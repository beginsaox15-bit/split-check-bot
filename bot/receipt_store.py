"""Local, versioned receipt snapshots. No FSM sessions, images or raw OCR data."""
import asyncio
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from bot.expense_split import Allocation
from bot.services.receipt_models import Money, ReceiptItem

MAX_BYTES = 256 * 1024
MAX_RECEIPTS = 100
SCHEMA_VERSION = 1


class StoreError(Exception):
    """Errors deliberately contain no SQLite messages, paths or receipt values."""


class StoreUnavailable(StoreError):
    pass


class ReceiptNotFound(StoreError):
    pass


class InvalidRecord(StoreError):
    pass


class VersionConflict(StoreError):
    pass


class ReceiptLimit(StoreError):
    pass


class ReceiptFacts(BaseModel):
    model_config = ConfigDict(extra='forbid')
    items: list[ReceiptItem] = Field(min_length=1, max_length=200)
    receipt_total: Money | None = None
    gross_total: Money | None = None
    prepayment: Money | None = None
    prepayment_ocr: Money | None = None
    prepayment_suspect: bool = False
    prepayment_verified: bool = False
    amount_due: Money | None = None
    derived_gross_total: Money | None = None
    guest_subtotals: dict[str, Money] = Field(default_factory=dict)
    currency: str | None = None
    ruble_evidence: bool = False


class ReceiptSnapshot(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal[1] = 1
    receipt: ReceiptFacts
    allocation: Allocation

    @model_validator(mode='after')
    def consistent_items(self):
        # The immutable source and allocation must describe the same lines.
        pairs = zip(self.receipt.items, self.allocation.items)
        if len(self.receipt.items) != len(self.allocation.items) or any(
                (a.name, a.quantity, a.total_price, a.guest_id) != (b.name, b.quantity, b.amount, b.guest_id)
                for a, b in pairs):
            raise ValueError('Snapshot items disagree')
        return self

    @classmethod
    def from_fsm(cls, data):
        plan = Allocation.model_validate(data['allocation'])
        facts = {key: data[key] for key in ReceiptFacts.model_fields if key in data}
        if len(data['items']) != len(plan.items):
            raise ValueError('Snapshot item count disagrees')
        items = []
        for source, item in zip(data['items'], plan.items):
            items.append({'name': item.name, 'quantity': item.quantity,
                          'unit_price': source.get('unit_price', source.get('price')),
                          'total_price': item.amount, 'guest_id': item.guest_id,
                          'suspicious': source.get('suspicious', False)})
        facts['items'] = items
        return cls(receipt=ReceiptFacts.model_validate(facts), allocation=plan)

    def fsm_data(self):
        return {**self.receipt.model_dump(mode='json'), 'allocation': self.allocation.model_dump(mode='json')}


class StoredReceipt(BaseModel):
    id: str = Field(pattern=r'^[0-9a-f]{32}$')
    owner_user_id: int = Field(gt=0)
    chat_id: int
    created_at: str
    updated_at: str
    state: Literal['allocation', 'payments']
    status: Literal['draft', 'completed']
    version: int = Field(ge=1)
    snapshot: ReceiptSnapshot

    @field_validator('created_at', 'updated_at')
    @classmethod
    def valid_timestamp(cls, value):
        if datetime.fromisoformat(value).tzinfo is None:
            raise ValueError('Timestamp must include timezone')
        return value


class ReceiptStore:
    def __init__(self, path, *, max_receipts=MAX_RECEIPTS):
        self.path = Path(path).expanduser()
        self.max_receipts = max_receipts
        self._initialized = False

    @contextmanager
    def connection(self):
        try:
            connection = sqlite3.connect(self.path, timeout=1)
            connection.row_factory = sqlite3.Row
            try:
                yield connection
            finally:
                connection.close()
        except sqlite3.Error:
            raise StoreUnavailable('Receipt storage is unavailable') from None

    async def initialize(self):
        await asyncio.to_thread(self._initialize)
        self._initialized = True

    def _initialize(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # Restrictive permissions from the first creation; never truncate a DB.
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(fd)
        except OSError:
            raise StoreUnavailable('Receipt storage is unavailable') from None
        with self.connection() as db:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise StoreUnavailable('Unsupported storage schema version')
            with db:
                db.execute('''CREATE TABLE IF NOT EXISTS receipts (
                    id TEXT PRIMARY KEY, owner_user_id INTEGER NOT NULL, chat_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('allocation', 'payments')),
                    status TEXT NOT NULL CHECK (status IN ('draft', 'completed')),
                    version INTEGER NOT NULL CHECK (version > 0),
                    data_version INTEGER NOT NULL, payload TEXT NOT NULL)''')
                db.execute('CREATE INDEX IF NOT EXISTS receipts_owner ON receipts(owner_user_id, chat_id, updated_at)')
                db.execute('PRAGMA user_version = 1')

    def _decode(self, row):
        try:
            if (row['data_version'] != 1 or not isinstance(row['payload'], str)
                    or len(row['payload'].encode('utf-8')) > MAX_BYTES):
                raise ValueError
            snap = ReceiptSnapshot.model_validate_json(row['payload'])
            plan = snap.allocation
            if (plan.owner_user_id != row['owner_user_id'] or plan.owner_chat_id != row['chat_id']
                    or row['status'] != ('completed' if plan.settlement_confirmed else 'draft')
                    or row['state'] != ('payments' if plan.completed else 'allocation')):
                raise ValueError
            return StoredReceipt.model_validate({**dict(row), 'snapshot': snap})
        except (ValidationError, ValueError, TypeError, KeyError):
            raise InvalidRecord('Receipt data is invalid') from None

    async def get(self, receipt_id, *, user_id, chat_id):
        if not self._initialized:
            await self.initialize()
        return await asyncio.to_thread(self._get, receipt_id, user_id, chat_id)

    def _get(self, receipt_id, user_id, chat_id):
        with self.connection() as db:
            row = db.execute('SELECT * FROM receipts WHERE id=? AND owner_user_id=? AND chat_id=?',
                             (receipt_id, user_id, chat_id)).fetchone()
        if row is None:
            raise ReceiptNotFound('Receipt not found')
        return self._decode(row)

    async def list(self, *, user_id, chat_id, offset=0, limit=10):
        if not self._initialized:
            await self.initialize()
        return await asyncio.to_thread(self._list, user_id, chat_id, offset, limit)

    def _list(self, user_id, chat_id, offset, limit):
        with self.connection() as db:
            rows = db.execute('SELECT * FROM receipts WHERE owner_user_id=? AND chat_id=? '
                              'ORDER BY updated_at DESC, id LIMIT ? OFFSET ?',
                              (user_id, chat_id, min(max(limit, 1), 11), max(offset, 0))).fetchall()
        # Corruption is isolated to one row and never logged or exposed.
        result = []
        for row in rows:
            try:
                result.append(self._decode(row))
            except InvalidRecord:
                result.append(None)
        return result

    async def save(self, snapshot, *, receipt_id=None, expected_version=0):
        if not self._initialized:
            await self.initialize()
        return await asyncio.to_thread(self._save, snapshot, receipt_id, expected_version)

    def _save(self, snapshot, receipt_id, expected_version):
        try:
            snap = ReceiptSnapshot.model_validate(snapshot.model_dump(warnings=False))
            payload = snap.model_dump_json()
        except (ValidationError, ValueError, TypeError):
            raise InvalidRecord('Receipt data is invalid') from None
        if len(payload.encode('utf-8')) > MAX_BYTES:
            raise ReceiptLimit('Receipt is too large')
        plan = snap.allocation
        receipt_id = receipt_id or uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        status = 'completed' if plan.settlement_confirmed else 'draft'
        state = 'payments' if plan.completed else 'allocation'
        with self.connection() as db, db:
            # Serialize quota checking/insertion and CAS updates in one transaction.
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM receipts WHERE id=? AND owner_user_id=? AND chat_id=?',
                             (receipt_id, plan.owner_user_id, plan.owner_chat_id)).fetchone()
            if row is None:
                if expected_version != 0:
                    raise VersionConflict('Receipt version changed')
                count = db.execute('SELECT COUNT(*) FROM receipts WHERE owner_user_id=?',
                                   (plan.owner_user_id,)).fetchone()[0]
                if count >= self.max_receipts:
                    raise ReceiptLimit('Saved receipt limit reached')
                if db.execute('SELECT 1 FROM receipts WHERE id=?', (receipt_id,)).fetchone():
                    raise VersionConflict('Receipt version changed')
                db.execute('INSERT INTO receipts VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1, ?)',
                           (receipt_id, plan.owner_user_id, plan.owner_chat_id, now, now, state, status, payload))
            else:
                previous = self._decode(row)
                # Equal repeated delivery is idempotent, including with the old version.
                if previous.snapshot == snap:
                    return previous
                if row['version'] != expected_version:
                    raise VersionConflict('Receipt version changed')
                db.execute('UPDATE receipts SET payload=?, state=?, status=?, updated_at=?, version=version+1 '
                           'WHERE id=? AND owner_user_id=? AND chat_id=? AND version=?',
                           (payload, state, status, now, receipt_id, plan.owner_user_id, plan.owner_chat_id, expected_version))
            row = db.execute('SELECT * FROM receipts WHERE id=?', (receipt_id,)).fetchone()
            result = self._decode(row)
        return result

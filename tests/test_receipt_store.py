"""Synthetic snapshots only; each test owns a temporary SQLite database."""
import asyncio
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from bot.expense_split import snapshot, add_participant
from bot.receipt_store import (ReceiptSnapshot, ReceiptStore, InvalidRecord, ReceiptNotFound,
                               ReceiptLimit, VersionConflict, StoreUnavailable)


def sample(user=42, chat=42):
    data = {'items': [{'name': 'Тестовая позиция', 'price': '10.00', 'quantity': 1}],
            'currency': 'RUB', 'receipt_total': '10.00'}
    data['allocation'] = snapshot(data, user_id=user, chat_id=chat).model_dump(mode='json')
    return ReceiptSnapshot.from_fsm(data)


class ReceiptStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'private/receipts.sqlite3'
        self.store = ReceiptStore(self.path)
        await self.store.initialize()

    async def test_create_read_update_and_decimal_strings(self):
        source = sample()
        saved = await self.store.save(source)
        read = await self.store.get(saved.id, user_id=42, chat_id=42)
        self.assertEqual(read, saved)
        self.assertEqual(read.version, 1)
        self.assertEqual(read.status, 'draft')
        source.allocation = add_participant(source.allocation, 'Участник')
        updated = await self.store.save(source, receipt_id=saved.id, expected_version=1)
        self.assertEqual(updated.version, 2)
        self.assertEqual(updated.created_at, saved.created_at)
        self.assertGreater(updated.updated_at, saved.updated_at)
        with sqlite3.connect(self.path) as db:
            payload = json.loads(db.execute('SELECT payload FROM receipts').fetchone()[0])
            self.assertEqual(payload['allocation']['items'][0]['amount'], '10.00')
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 1)

    async def test_users_chats_and_parameterized_ids(self):
        a, b = await self.store.save(sample()), await self.store.save(sample(43, 43))
        self.assertEqual([x.id for x in await self.store.list(user_id=42, chat_id=42)], [a.id])
        self.assertEqual([x.id for x in await self.store.list(user_id=43, chat_id=43)], [b.id])
        for user, chat, key in [(43, 42, a.id), (42, 43, a.id), (42, 42, "' OR 1=1 --")]:
            with self.assertRaises(ReceiptNotFound):
                await self.store.get(key, user_id=user, chat_id=chat)
        with self.assertRaises(VersionConflict):
            await self.store.save(sample(43, 43), receipt_id=a.id, expected_version=0)

    async def test_idempotent_delivery_preserves_version_and_timestamp(self):
        saved = await self.store.save(sample())
        for version in (0, 1):
            again = await self.store.save(sample(), receipt_id=saved.id, expected_version=version)
            self.assertEqual(again, saved)
        self.assertEqual(len(await self.store.list(user_id=42, chat_id=42)), 1)

    async def test_conflict_does_not_overwrite(self):
        saved = await self.store.save(sample())
        one, two = sample(), sample()
        one.allocation = add_participant(one.allocation, 'Первый')
        two.allocation = add_participant(two.allocation, 'Второй')
        await self.store.save(one, receipt_id=saved.id, expected_version=1)
        with self.assertRaises(VersionConflict):
            await self.store.save(two, receipt_id=saved.id, expected_version=1)
        self.assertEqual((await self.store.get(saved.id, user_id=42, chat_id=42)).snapshot, one)

    async def test_concurrent_cas_only_one_writer_wins(self):
        saved = await self.store.save(sample())
        versions = [sample(), sample()]
        for i, value in enumerate(versions):
            value.allocation = add_participant(value.allocation, f'Участник {i}')
        results = await asyncio.gather(*(self.store.save(value, receipt_id=saved.id, expected_version=1)
                                         for value in versions), return_exceptions=True)
        self.assertEqual(sum(isinstance(x, VersionConflict) for x in results), 1)
        self.assertEqual((await self.store.get(saved.id, user_id=42, chat_id=42)).version, 2)

    async def test_older_record_gets_safe_defaults_and_restart(self):
        saved = await self.store.save(sample())
        with sqlite3.connect(self.path) as db:
            payload = json.loads(db.execute('SELECT payload FROM receipts').fetchone()[0])
            for key in ('tip_amount', 'tip_payer_id', 'discount_amount', 'payments', 'settlement_confirmed'):
                payload['allocation'].pop(key)
            db.execute('UPDATE receipts SET payload=?', (json.dumps(payload),))
        restarted = ReceiptStore(self.path)
        await restarted.initialize()
        record = await restarted.get(saved.id, user_id=42, chat_id=42)
        self.assertEqual(record.snapshot.allocation.tip_amount, '0.00')
        self.assertEqual(record.snapshot.allocation.payments, {})

    async def test_corruption_isolated_and_never_contains_raw_values_in_error(self):
        saved = await self.store.save(sample())
        healthy = await self.store.save(sample())
        with sqlite3.connect(self.path) as db:
            db.execute('UPDATE receipts SET payload=? WHERE id=?', ('PRIVATE BROKEN JSON', saved.id))
        with self.assertRaises(InvalidRecord) as error:
            await self.store.get(saved.id, user_id=42, chat_id=42)
        self.assertNotIn('PRIVATE', str(error.exception))
        records = await self.store.list(user_id=42, chat_id=42)
        self.assertIn(None, records)
        self.assertIn(healthy.id, [r.id for r in records if r])

    async def test_invalid_money_is_rejected_without_writing(self):
        for value in (1.1, 'NaN', '-1', '1.001', '1e2', '0.1.0'):
            source = sample()
            source.allocation.items[0].amount = value
            with self.subTest(value=value), self.assertRaises(InvalidRecord):
                await self.store.save(source)
        self.assertEqual(await self.store.list(user_id=42, chat_id=42), [])

    async def test_invalid_state_owner_status_and_data_version_are_rejected(self):
        saved = await self.store.save(sample())
        for field, value in [('owner_user_id', 43), ('status', 'completed'), ('data_version', 2), ('state', 'payments'), ('created_at', 'corrupted')]:
            with sqlite3.connect(self.path) as db:
                original = db.execute(f'SELECT {field} FROM receipts').fetchone()[0]
                db.execute(f'UPDATE receipts SET {field}=?', (value,))
            with self.assertRaises(InvalidRecord):
                # Owner mismatch still passes SQL scope to exercise payload ownership.
                await self.store.get(saved.id, user_id=43 if field == 'owner_user_id' else 42, chat_id=42)
            with sqlite3.connect(self.path) as db:
                db.execute(f'UPDATE receipts SET {field}=?', (original,))

    async def test_quota_has_no_automatic_deletion_and_allows_updates(self):
        self.store.max_receipts = 1
        saved = await self.store.save(sample())
        with self.assertRaises(ReceiptLimit):
            await self.store.save(sample())
        await self.store.save(sample(43, 43))
        changed = sample()
        changed.allocation = add_participant(changed.allocation, 'Участник')
        await self.store.save(changed, receipt_id=saved.id, expected_version=1)
        self.assertEqual(len(await self.store.list(user_id=42, chat_id=42)), 1)

    async def test_payload_size_is_bounded(self):
        source = sample()
        source.allocation.items[0].guest_id = 'x' * (256*1024)
        source.receipt.items[0].guest_id = source.allocation.items[0].guest_id
        with self.assertRaises(ReceiptLimit):
            await self.store.save(source)

    async def test_only_allowlisted_facts_no_raw_ocr_image_or_session(self):
        source = sample()
        data = source.fsm_data()
        data.update(raw_text='PRIVATE', receipt_photo={'file_id': 'PRIVATE'}, recognition={'raw_text': 'PRIVATE'},
                    allocation_token='PRIVATE', payment_input={'token': 'PRIVATE'}, BOT_TOKEN='PRIVATE',
                    OPENROUTER_API_KEY='PRIVATE', base64='PRIVATE')
        saved = await self.store.save(ReceiptSnapshot.from_fsm(data))
        self.assertNotIn('PRIVATE', saved.snapshot.model_dump_json())
        self.assertNotIn('allocation_token', saved.snapshot.fsm_data())

    async def test_locked_database_fails_safely_then_recovers(self):
        with sqlite3.connect(self.path) as db:
            db.execute('BEGIN EXCLUSIVE')
            with self.assertRaises(StoreUnavailable):
                await self.store.save(sample())
        saved = await self.store.save(sample())
        self.assertEqual(saved.version, 1)

    async def test_sqlite_aborted_write_leaves_previous_version_intact(self):
        saved = await self.store.save(sample())
        changed = sample()
        changed.allocation = add_participant(changed.allocation, 'Участник')
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TRIGGER reject_update BEFORE UPDATE ON receipts BEGIN SELECT RAISE(ABORT, 'private'); END")
        with self.assertRaises(StoreUnavailable) as error:
            await self.store.save(changed, receipt_id=saved.id, expected_version=1)
        self.assertNotIn('private', str(error.exception))
        self.assertEqual(await self.store.get(saved.id, user_id=42, chat_id=42), saved)

    async def test_facts_keep_original_totals_prepayment_and_unit_prices(self):
        data = sample().fsm_data()
        data.update(gross_total='10.00', prepayment=None, prepayment_ocr='4.01',
                    prepayment_suspect=True, amount_due='6.00', guest_subtotals={'1': '10.00'})
        data['items'][0]['guest_id'] = '1'
        data['allocation'].update(gross_total='10.00', prepayment=None, amount_due='6.00',
                                  prepayment_suspect=True)
        data['allocation']['items'][0]['guest_id'] = '1'
        snap = ReceiptSnapshot.from_fsm(data)
        record = await self.store.save(snap)
        restored = (await self.store.get(record.id, user_id=42, chat_id=42)).snapshot.fsm_data()
        self.assertIsNone(restored['prepayment'])
        self.assertEqual(restored['prepayment_ocr'], '4.01')
        self.assertEqual(restored['items'][0]['unit_price'], '10.00')
        self.assertEqual(restored['items'][0]['guest_id'], '1')
        self.assertEqual(restored['guest_subtotals'], {'1': '10.00'})

    async def test_failed_startup_initialization_can_recover_on_next_operation(self):
        uninitialized = ReceiptStore(self.path.parent / 'new.sqlite3')
        with patch.object(uninitialized, '_initialize', side_effect=StoreUnavailable('offline')):
            with self.assertRaises(StoreUnavailable):
                await uninitialized.initialize()
        saved = await uninitialized.save(sample())
        self.assertEqual(saved.version, 1)

    async def test_newer_database_schema_is_not_downgraded(self):
        with sqlite3.connect(self.path) as db:
            db.execute('PRAGMA user_version=99')
        with self.assertRaises(StoreUnavailable):
            await self.store.initialize()
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 99)

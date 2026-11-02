"""Длинный узкий русский чек, только синтетические товары без персональных данных."""
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent

def main():
    rows = ['ЧЕК (ТЕСТОВЫЙ)', 'Наименование Кол-во Сумма']
    items = []
    groups = [('1', 40, '275.00', '425.00', '11150.00'),
              ('2', 100, '290.00', '372.50', '29082.50'),
              ('3', 10, '167.00', '167.00', '1670.00')]
    names = ['Тартар из говядины', 'Сырная тарелка', 'Пастрами сэндвич', 'Лимонад домашний']
    for guest, count, price, last, total in groups:
        rows.append('Гость ' + guest)
        for i in range(count):
            name, amount = names[i % len(names)], last if i == count - 1 else price
            rows.append(f'{name} 1 {amount}')
            items.append({'name': name, 'quantity': '1', 'total_price': amount})
        rows.append(f'Итого к оплате Гость {guest}: {total}')
    rows += ['Полная сумма 41902.50', 'Предоплата 7000.00', 'Итого к оплате 34902.50']
    with Image.new('RGB', (640, 70 + 40 * len(rows)), (232, 232, 232)) as image:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(os.getenv('TESSERACT_SMOKE_FONT', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'), 18)
        for i, row in enumerate(rows):
            draw.text((25, 25 + 40 * i), row, font=font, fill=(65, 65, 65))
        image.save(ROOT / 'long_restaurant_dense.png')
    data = {'name':'long_restaurant_dense','image':'long_restaurant_dense.png',
            'tags':['long','restaurant','guests','duplicates','small_text'], 'expected_items':items,
            'expected_total':'41902.50','currency':'RUB','expected_gross_total':'41902.50',
            'expected_prepayment':'7000.00','expected_amount_due':'34902.50',
            'expected_guest_subtotals':{guest:total for guest,_,_,_,total in groups}}
    (ROOT / 'expected/long_restaurant_dense.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

if __name__ == '__main__':
    main()

import os
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile
from aiogram.fsm.storage.memory import MemoryStorage

# =====================
# 1. Загрузка моделей
# =====================
model_yolo = YOLO('"C:\нейронка пропуски\best.pt"')  # Ваш файл best.pt
reader = easyocr.Reader(['en'], gpu=False)

def preprocess_digits(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

async def infer_img(image_path: str):
    results = model_yolo(image_path)
    img = cv2.imread(image_path)

    herb_found = False
    number_bbox = None
    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = model_yolo.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        if cls_name == 'spbu_herb':
            herb_found = True
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        elif cls_name == 'dorm_number':
            number_bbox = (x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = 'SPbU Pass' if herb_found else 'Other Card'
    cv2.putText(
        img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )

    ocr_text = '—'
    if herb_found and number_bbox:
        x1, y1, x2, y2 = number_bbox
        pad = 5
        roi = img[
            max(0, y1-pad):min(img.shape[0], y2+pad),
            max(0, x1-pad):min(img.shape[1], x2+pad)
        ]
        proc = preprocess_digits(roi)
        res = reader.readtext(proc, allowlist='0123456789', detail=0)
        ocr_text = ''.join(res) if res else 'Не распознано'
        cv2.putText(
            img,
            f'ID: {ocr_text}',
            (x1, y2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    base = os.path.basename(image_path)
    out_dir = 'tmp'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'result_{base}')
    cv2.imwrite(out_path, img)
    return out_path, ocr_text

# =====================
# 2. Настройка бота на aiogram3
# =====================
BOT_TOKEN = '8124300044:AAFJ32OS7UREoXiWUUlke8YOkC5W2Mf8RX8'
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

@dp.message(Command(commands=['start']))
async def cmd_start(message: Message):
    await message.reply(
        "Привет! Пришли фото карты — я определю SPbU пропуск и извлеку номер."
    )

@dp.message(lambda msg: msg.photo)
async def handle_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    # Скачиваем файл в BytesIO
    file_bytes_io = await bot.download_file(file.file_path)
    tmp_dir = 'tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    input_path = os.path.join(tmp_dir, f'{photo.file_id}.jpg')
    # Сохраняем в локальный файл
    with open(input_path, 'wb') as f:
        f.write(file_bytes_io.getvalue())

    annotated_path, ocr_text = await infer_img(input_path)

    # Отправляем аннотированное изображение
    photo_to_send = FSInputFile(annotated_path)
    await bot.send_photo(
    chat_id=message.chat.id,
    photo=photo_to_send,
    caption=f"Распознанный номер: {ocr_text}"
)

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())

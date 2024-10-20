import logging
import asyncio
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import random

from aiogram import Bot, Dispatcher, types
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup
from aiogram.utils import executor
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove
from aiogram.contrib.fsm_storage.memory import MemoryStorage
#from catboost import CatBoostClassifier


from aiogram.contrib.middlewares.logging import LoggingMiddleware

HELP_COMMAND = '''
/start - начать работу
/help - список команд
'''
FILE_PATH = 'user.parquet'  # Укажите путь к вашему файлу user.parquet
MODEL_PATH = 'model.pkl'  # Укажите путь к вашему файлу модели
from config import API_TOKEN
storage = MemoryStorage()

logging.basicConfig(level=logging.INFO)
class Form(StatesGroup):
    waiting_for_user_id = State()
# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
button_start = KeyboardButton("/start")
button_help = KeyboardButton("/help")

model_type = None

# Главное меню
menu_keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
menu_keyboard.add(KeyboardButton('Статистика модели'), KeyboardButton('Старт сессии'))

main_menu = ReplyKeyboardMarkup(resize_keyboard=True)
main_menu.add(KeyboardButton('Главное меню'))

# Клавиатура для проверки
check_keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
check_keyboard.row(KeyboardButton('Проверка по ID пользователя'), KeyboardButton('Проверка по данным'))
check_keyboard.add(KeyboardButton('Главное меню'))

statistics_keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
statistics_keyboard.row(KeyboardButton("Экстраверт - Интроверт"), KeyboardButton("Аналитик - Чувствительный"), KeyboardButton("Ригидный - Гибкий"))
statistics_keyboard.add(KeyboardButton('Главное меню'))

model_keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
model_keyboard.add(KeyboardButton('K-means'), KeyboardButton('Языковая модель'))

@dp.message_handler(lambda message: message.text == "Статистика модели")
async def show_statistics(message: types.Message):
    await message.answer("Графики K-means распределения из Датасета будут здесь.")
    await message.answer("Выберите категорию:", reply_markup=statistics_keyboard)

@dp.message_handler(lambda message: message.text == "Экстраверт - Интроверт")
async def show_extravert_introvert(message: types.Message):
    with open("extravert_introvert_chart.jpg", 'rb') as photo:
        await message.answer_photo(photo=photo, caption="График распределения Экстраверт - Интроверт.", reply_markup=menu_keyboard)

@dp.message_handler(lambda message: message.text == "Аналитик - Чувствительный")
async def show_analytic_sensitive(message: types.Message):
    with open("analytic_sensitive_chart.jpg", 'rb') as photo:
        await message.answer_photo(photo=photo, caption="График распределения Аналитик - Чувствительный.", reply_markup=menu_keyboard)

@dp.message_handler(lambda message: message.text == "Ригидный - Гибкий")
async def show_rigid_flexible(message: types.Message):
    with open("rigid_flexible_chart.jpg", 'rb') as photo:
        await message.answer_photo(photo=photo, caption="График распределения Ригидный - Гибкий.", reply_markup=menu_keyboard)

@dp.message_handler(lambda message: message.text == "Старт сессии")
async def start_session(message: types.Message):
    await message.answer("Выберите метод проверки:", reply_markup=check_keyboard)

@dp.message_handler(lambda message: message.text == "Главное меню")
async def back_to_menu(message: types.Message):
    await message.answer("Возвращение в главное меню.", reply_markup=menu_keyboard)

@dp.message_handler(lambda message: message.text == "Проверка по ID пользователя")
async def check_by_id(message: types.Message):
    await message.answer("Выберите модель для предсказания:", reply_markup=model_keyboard)

@dp.message_handler(lambda message: message.text == "K-means")
async def select_kmeans(message: types.Message):
    global model_type  # Используем глобальную переменную
    model_type = 'kmeans'
    await message.answer("Введите ID пользователя (только числовой формат):", reply_markup=ReplyKeyboardRemove())

@dp.message_handler(lambda message: message.text == "Языковая модель")
async def select_llm(message: types.Message):
    global model_type  # Используем глобальную переменную
    model_type = 'llm'
    await message.answer("Введите ID пользователя (только числовой формат):", reply_markup=ReplyKeyboardRemove())

@dp.message_handler(lambda message: message.text == "Проверка по данным")
async def check_by_data(message: types.Message):
    await message.answer("Введите данные:", reply_markup=ReplyKeyboardRemove())

@dp.message_handler(lambda message: message.text.isdigit())
async def process_id(message: types.Message):
    global model_type  # Используем глобальную переменную
    vk_id = message.text
    try:
        if model_type == 'kmeans':
            # Читаем данные из файла submit.parquet
            df = pd.read_parquet('submit.parquet')

            # Получаем значения для указанного vk_id
            user_row = df.loc[df['vk_id'] == int(vk_id)]

            # Проверяем, существует ли пользователь с указанным vk_id
            if not user_row.empty:
                combined_target = user_row['combined_target'].values[0]
                result_message = f"Психотип пользователя: {combined_target}."
            else:
                result_message = f"Пользователь с ID: {vk_id} не найден."

            await message.answer(result_message, reply_markup=main_menu)

        elif model_type == 'llm':
            df_llm = pd.read_parquet('df_llm.parquet')
            user_data = df_llm[df_llm['vk_id'] == str(vk_id)]

            if not user_data.empty:
                is_extravert = user_data.iloc[0]['is_extravert']
                is_analyst = user_data.iloc[0]['is_analyst']
                is_rigid = user_data.iloc[0]['is_rigid']

                type_person = ""
                type_person += "Э" if is_extravert else "И"
                type_person += "А" if is_analyst else "Ч"
                type_person += "Р" if is_rigid else "Г"

                await message.answer(f"Психотип пользователя: 1{type_person}.", reply_markup=main_menu)
            else:
                await message.answer(f"Аккаунт с ID {vk_id} не найден.", reply_markup=main_menu)

    except FileNotFoundError:
        await message.answer(f"Файл не найден. Убедитесь, что файл загружен правильно.")
    except IndexError:
        await message.answer(f"ID пользователя не найден в данных.")
    except Exception as e:
        await message.answer(f"Произошла ошибка: {str(e)}")



# # Обработка ввода данных после проверки по данным
# @dp.message_handler(lambda message: message.text != "Статистика модели" and message.text != "Старт сессии" and message.text != "Проверка по ID пользователя" and message.text != "Проверка по данным")
# async def process_data(message: types.Message):
#     user_data = message.text
#     # Здесь обработка введенных данных через модель и предсказание
#     predicted_type = "Психотип: A"  # Замените на реальный результат предсказания
#     await message.answer(f"Ваш психотип: {predicted_type}")
#     await message.answer("Возвращение в главное меню", reply_markup=menu_keyboard)



@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.answer("Добро пожаловать! Выберите действие:", reply_markup=menu_keyboard)



# Обработчик команды /help
@dp.message_handler(commands=['help'])
async def help_command(message: types.Message):
    await message.reply(text=HELP_COMMAND)

@dp.message_handler()
async def unknown_command(message: types.Message):
    await message.answer("Неизвестная команда")



if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

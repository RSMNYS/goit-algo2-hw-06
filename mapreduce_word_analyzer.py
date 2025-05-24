#!/usr/bin/env python3
"""
MapReduce Word Frequency Analyzer

Скрипт для аналізу частоти використання слів у тексті за допомогою парадигми MapReduce
з візуалізацією результатів.

Автор: [Ваше ПІБ]
Дата: 2025
"""

import re
import string
import requests
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict
import time


class MapReduceWordCounter:
    """Клас для реалізації MapReduce обчислення частоти слів."""
    
    def __init__(self, num_workers: int = 4):
        """
        Ініціалізація MapReduce процесора.
        
        Args:
            num_workers: кількість робочих потоків
        """
        self.num_workers = num_workers
    
    def clean_text(self, text: str) -> str:
        """
        Очищує текст від пунктуації та приводить до нижнього регістру.
        
        Args:
            text: вхідний текст
            
        Returns:
            очищений текст
        """
        # Видаляємо пунктуацію та цифри
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        # Приводимо до нижнього регістру
        text = text.lower()
        # Видаляємо зайві пробіли
        text = ' '.join(text.split())
        return text
    
    def split_text(self, text: str, num_chunks: int) -> List[str]:
        """
        Розділяє текст на частини для паралельної обробки.
        
        Args:
            text: вхідний текст
            num_chunks: кількість частин
            
        Returns:
            список частин тексту
        """
        words = text.split()
        chunk_size = len(words) // num_chunks
        
        chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            if i == num_chunks - 1:  # остання частина включає залишок
                end_idx = len(words)
            else:
                end_idx = (i + 1) * chunk_size
            
            chunks.append(' '.join(words[start_idx:end_idx]))
        
        return chunks
    
    def map_function(self, text_chunk: str) -> List[Tuple[str, int]]:
        """
        Map функція: перетворює частину тексту в пари (слово, 1).
        
        Args:
            text_chunk: частина тексту
            
        Returns:
            список пар (слово, 1)
        """
        words = text_chunk.split()
        # Фільтруємо короткі слова та стоп-слова
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 
                     'into', 'through', 'during', 'before', 'after', 'above', 
                     'below', 'between', 'among', 'is', 'are', 'was', 'were', 
                     'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
                     'did', 'will', 'would', 'should', 'could', 'can', 'may', 
                     'might', 'must', 'shall', 'i', 'you', 'he', 'she', 'it', 
                     'we', 'they', 'them', 'their', 'this', 'that', 'these', 
                     'those', 'not', 'no', 'yes', 'all', 'any', 'some', 'each', 
                     'every', 'most', 'more', 'less', 'much', 'many', 'few', 
                     'little', 'big', 'small', 'large', 'great', 'good', 'bad', 
                     'new', 'old', 'first', 'last', 'next', 'same', 'other'}
        
        word_pairs = []
        for word in words:
            # Видаляємо слова коротші за 3 символи та стоп-слова
            if len(word) >= 3 and word not in stop_words:
                word_pairs.append((word, 1))
        
        return word_pairs
    
    def reduce_function(self, word_counts: List[Tuple[str, int]]) -> Dict[str, int]:
        """
        Reduce функція: агрегує підрахунки для кожного слова.
        
        Args:
            word_counts: список пар (слово, кількість)
            
        Returns:
            словник з підрахунками слів
        """
        word_freq = defaultdict(int)
        for word, count in word_counts:
            word_freq[word] += count
        return dict(word_freq)
    
    def mapreduce(self, text: str) -> Dict[str, int]:
        """
        Виконує MapReduce для підрахунку частоти слів.
        
        Args:
            text: вхідний текст
            
        Returns:
            словник з частотою слів
        """
        print("Початок MapReduce обробки...")
        start_time = time.time()
        
        # Очищуємо текст
        cleaned_text = self.clean_text(text)
        print(f"Текст очищено. Кількість слів: {len(cleaned_text.split())}")
        
        # Розділяємо текст на частини
        text_chunks = self.split_text(cleaned_text, self.num_workers)
        print(f"Текст розділено на {len(text_chunks)} частин")
        
        # Map фаза (паралельна)
        all_word_pairs = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Подаємо завдання на виконання
            map_futures = [executor.submit(self.map_function, chunk) 
                          for chunk in text_chunks]
            
            # Збираємо результати
            for future in as_completed(map_futures):
                word_pairs = future.result()
                all_word_pairs.extend(word_pairs)
        
        print(f"Map фаза завершена. Знайдено {len(all_word_pairs)} пар слів")
        
        # Reduce фаза
        word_frequencies = self.reduce_function(all_word_pairs)
        
        end_time = time.time()
        print(f"MapReduce завершено за {end_time - start_time:.2f} секунд")
        print(f"Знайдено {len(word_frequencies)} унікальних слів")
        
        return word_frequencies


def download_text(url: str) -> str:
    """
    Завантажує текст з URL-адреси.
    
    Args:
        url: URL-адреса тексту
        
    Returns:
        завантажений текст
    """
    try:
        print(f"Завантаження тексту з {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        print(f"Текст завантажено. Розмір: {len(response.text)} символів")
        return response.text
    except requests.RequestException as e:
        print(f"Помилка завантаження: {e}")
        # Використовуємо приклад тексту для демонстрації
        return """
        This is a sample text for demonstration purposes. This text contains various words 
        that will be analyzed for frequency. The word frequency analysis will show which 
        words appear most often in this text. Some words like 'the', 'and', 'is' are 
        common words that appear frequently in English text. Other words might be more 
        specific to the content of the text. The MapReduce paradigm allows us to process 
        large amounts of text efficiently by dividing the work among multiple processes.
        """ * 50  # Повторюємо для більшого обсягу


def visualize_top_words(word_frequencies: Dict[str, int], top_n: int = 10):
    """
    Візуалізує топ-слова з найвищою частотою.
    
    Args:
        word_frequencies: словник з частотою слів
        top_n: кількість топ-слів для відображення
    """
    # Сортуємо слова за частотою
    sorted_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
    top_words = sorted_words[:top_n]
    
    if not top_words:
        print("Немає слів для візуалізації")
        return
    
    # Підготовка даних для графіка
    words = [word for word, _ in top_words]
    frequencies = [freq for _, freq in top_words]
    
    # Створення графіка
    plt.figure(figsize=(12, 8))
    bars = plt.barh(words, frequencies, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Налаштування графіка
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Words', fontsize=12)
    plt.title(f'Top {top_n} Most Frequent Words', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Інвертуємо y-осі для кращого відображення
    
    # Додаємо значення на стовпці
    for bar, freq in zip(bars, frequencies):
        plt.text(bar.get_width() + max(frequencies) * 0.01, 
                bar.get_y() + bar.get_height()/2, 
                str(freq), ha='left', va='center', fontweight='bold')
    
    # Покращуємо макет
    plt.tight_layout()
    plt.grid(axis='x', alpha=0.3)
    
    # Показуємо графік
    plt.show()
    
    # Виводимо результати в консоль
    print(f"\nТоп {top_n} найчастіших слів:")
    print("-" * 30)
    for i, (word, freq) in enumerate(top_words, 1):
        print(f"{i:2d}. {word:<15} {freq:>5}")


def main():
    """Головна функція програми."""
    print("=" * 50)
    print("MapReduce Word Frequency Analyzer")
    print("=" * 50)
    
    url = "https://www.gutenberg.org/files/11/11-0.txt"

    try:
        # Завантажуємо текст
        text = download_text(url)
        
        # Створюємо MapReduce процесор
        mr_processor = MapReduceWordCounter(num_workers=4)
        
        # Виконуємо аналіз частоти слів
        word_frequencies = mr_processor.mapreduce(text)
        
        # Візуалізуємо результати
        visualize_top_words(word_frequencies, top_n=15)
        
        # Додаткова статистика
        print(f"\nЗагальна статистика:")
        print(f"Унікальних слів: {len(word_frequencies)}")
        print(f"Загальна кількість слів: {sum(word_frequencies.values())}")
        
        # Найрідші слова
        rare_words = [word for word, freq in word_frequencies.items() if freq == 1]
        print(f"Слова, що зустрічаються один раз: {len(rare_words)}")
        
    except Exception as e:
        print(f"Помилка виконання програми: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
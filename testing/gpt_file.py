import os

# Укажи нужный порядок файлов и их типы для лучшей структуры
files = [
    ('describe_image.py', 'python'),
    ('index_builder.py', 'python'),
    ('prompts_config.py', 'python'),
    ('reasoning_engine.py', 'python'),
    ('run_demo.py', 'python'),
    ('search_system.py', 'python'),
    ('text_processor.py', 'python'),
]

output_filename = "merged_for_gpt.txt"

with open(output_filename, "w", encoding="utf-8") as outfile:
    for filename, filetype in files:
        if not os.path.exists(filename):
            print(f"❗ Файл не найден: {filename}, пропущен")
            continue
        outfile.write(f"\n\n========== [ {filename} ] ({filetype.upper()}) ==========\n\n")
        with open(filename, "r", encoding="utf-8") as infile:
            outfile.write(infile.read())
            outfile.write("\n")  # На всякий случай — чтобы всегда был перенос строки

print(f"✅ Готово! Все файлы объединены в {output_filename}")

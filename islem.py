# Hedef kelimeler
keywords = ["python", "düzenle", "kopyala"]

# Dosyayı oku
with open("read.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# Satırları filtrele
filtered_lines = [line for line in lines if not any(kw in line.lower() for kw in keywords)]

# Sonuçları yazdır (istersen dosyaya da yazabilirsin)
for line in filtered_lines:
    print(line.strip())
with open("read.txt", "w", encoding="utf-8") as file:
    lines = file.readlines()
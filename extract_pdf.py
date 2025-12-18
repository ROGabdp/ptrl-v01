import fitz

doc = fitz.open('1-s2.0-S0957417424013319-main.pdf')
full_text = ""
for page_num, page in enumerate(doc):
    full_text += f"\n\n--- Page {page_num + 1} ---\n\n"
    full_text += page.get_text()

# Write to file
with open('pdf_content.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)

print(f"Total pages: {len(doc)}")
print(f"Total characters: {len(full_text)}")
print("Content saved to pdf_content.txt")

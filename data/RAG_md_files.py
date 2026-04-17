import re

def count_tokens(text):
    # این یک تابع نمونه برای شمارش توکن است.
    # در پیاده‌سازی واقعی، باید از یک توکن‌شمار مناسب مانند tiktoken استفاده کنید.
    return len(text.split())

def process_md_entries(entries, max_tokens=250, overlap=50):
    processed_docs = []
    
    # RecursiveCharacterTextSplitter برای متون طولانی مناسب است، اما برای این مورد که
    # می‌خواهیم هر بخش "**" به عنوان یک واحد در نظر گرفته شود،
    # پردازش دستی با regex مناسب‌تر است.
    # اگر متون ورودی بسیار طولانی هستند و نیاز به تقسیم‌بندی دقیق‌تر دارید،
    # می‌توان از RecursiveCharacterTextSplitter در کنار regex استفاده کرد.

    for entry in entries:
        content = entry.get("data", "")
        
        # استخراج عنوان اصلی (هدر با #)
        header_match = re.search(r'^#\s*(.*)', content, re.MULTILINE)
        main_title = header_match.group(1).strip() if header_match else "بدون عنوان"
        
        # جدا کردن بخش‌های مختلف بر اساس جملات شروع شونده با **
        # این regex به دنبال ** هر چیزی ** می‌گردد و محتوای بعد از آن را تا ** بعدی یا انتهای متن می‌گیرد.
        # استفاده از re.DOTALL برای اینکه '.' شامل کاراکترهای خط جدید هم بشود.
        sections = re.findall(r'\*\*(.*?)\*\*\s*(.*?)(?=\n\*\*|\Z)', content, re.DOTALL | re.MULTILINE)
        
        for title_part, data_part in sections:
            full_text = f"**{title_part.strip()}**\n{data_part.strip()}"
            media_tags = re.findall(r'!\[.*?\]\(.*?\)', data_part)
            if not media_tags:
        
                # در اینجا می‌توان token count را برای اطمینان از max_tokens بررسی کرد
                # اگرچه با توجه به اینکه هر chunk یک بخش "**" است، معمولاً از max_tokens تجاوز نمی‌کند
                # مگر اینکه توضیحات زیر هر بخش بسیار طولانی باشد.
                token_count = count_tokens(full_text)

                if token_count <= max_tokens:
                    processed_docs.append({
                        "text": full_text,
                        "metadata": {"title": main_title, "source_section": title_part.strip()}
                    })
                else:
                    # اگر chunk از max_tokens بیشتر بود، می‌توان آن را با text_splitter تقسیم کرد
                    # اما با توجه به ساختار markdown، این اتفاق کمتر رخ می‌دهد.
                    # برای سادگی، فعلا chunkهای بزرگ را هم اضافه می‌کنیم.
                    # در صورت نیاز، منطق تقسیم‌بندی chunkهای بزرگ به صورت جداگانه پیاده‌سازی شود.
                    processed_docs.append({
                        "text": full_text,
                        "metadata": {"title": main_title, "source_section": title_part.strip()}
                    })
                        
    return processed_docs

# مثال نحوه استفاده:
# فرض کنید 'entries' لیستی از دیکشنری‌ها مانند مثال شماست
with open('/Users/korosh/Desktop/ivs/data/MarkDown/Catalog.md', 'r', encoding='utf-8') as f:
    content = f.read()
print(content)
processed_chunks = process_md_entries([{'data':content}])
print(processed_chunks)

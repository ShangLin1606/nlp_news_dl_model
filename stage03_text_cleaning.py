from pathlib import Path
from tqdm import tqdm
import re


input_paths = ['data/traditional_train', 'data/traditional_test']
output_paths = ['data/traditional_train_clean', 'data/traditional_test_clean']

for input_path, output_path in zip(input_paths, output_paths):
    for file in tqdm(Path(input_path).glob('*/*.txt'), desc=f'Processing {input_path}'):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                text = re.sub(r'[ \t]+', ' ', text)
                cleaned_text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])

            output_file = Path(output_path) / file.relative_to(input_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

        except Exception as e:
            print(f"Error processing {file}: {e}")

print("文本清理並保存至指定資料夾完成！")

import os
import re
import argparse
from typing import List, Optional
import sys
from pathlib import Path

from config import config

class TextCleaner:
    """文本清洗器，用于清洗文本文件"""
    
    def __init__(self, input_dir: str = "", output_dir: str = ""):
        """初始化文本清洗器
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir or config.DATASET_PATH
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def clean_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """清洗单个文件
        
        Args:
            file_path: 文件路径
            output_path: 输出路径，如果为None则使用默认输出路径
            
        Returns:
            清洗后的文本
        """
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 清洗文本
        cleaned_text = self._clean_text(content)
        
        # 如果指定了输出路径，则保存文件
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)
        
        return cleaned_text
    
    def _clean_text(self, text: str) -> str:
        """清洗文本
        
        Args:
            text: 待清洗文本
            
        Returns:
            清洗后的文本
        """
        # 保留换行符，只处理每一行
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 移除行内多余的空白字符（保留单个空格）
            line = re.sub(r'\s+', ' ', line)
            
            # 移除特殊字符，但保留Markdown格式
            # 保留字母、数字、中文、常见标点和Markdown符号
            line = re.sub(r'[^\w\s\u4e00-\u9fff.,?!;:，。？！；：""''()#*\-_`~\[\]{}]', '', line)
            
            # 移除重复的标点符号
            line = re.sub(r'([.,?!;:，。？！；：])\1+', r'\1', line)
            
            cleaned_lines.append(line.strip())
        
        # 重新组合文本，保留换行符
        return '\n'.join(cleaned_lines)
    
    def clean_directory(self, file_extensions: List[str] = ['.txt']) -> List[str]:
        """清洗目录下的所有文件
        
        Args:
            file_extensions: 文件扩展名列表
            
        Returns:
            清洗后的文件路径列表
        """
        cleaned_files = []
        
        # 遍历输入目录中的所有文件
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                # 检查文件扩展名
                if any(file.endswith(ext) for ext in file_extensions):
                    # 构建文件路径
                    file_path = os.path.join(root, file)
                    
                    # 构建相对路径
                    rel_path = os.path.relpath(file_path, self.input_dir)
                    
                    # 构建输出路径
                    output_path = os.path.join(self.output_dir, rel_path)
                    
                    # 确保输出目录存在
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # 清洗文件
                    self.clean_file(file_path, output_path)
                    
                    # 添加到清洗后的文件列表
                    cleaned_files.append(output_path)
                    
                    print(f"已清洗: {file_path} -> {output_path}")
        
        return cleaned_files

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='文本清洗工具')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--extensions', type=str, nargs='+', default=['.txt'], help='文件扩展名列表')
    
    args = parser.parse_args()
    
    # 创建文本清洗器
    cleaner = TextCleaner(args.input_dir, args.output_dir or config.DATASET_PATH)
    
    # 清洗目录
    cleaned_files = cleaner.clean_directory(args.extensions)
    
    print(f"共清洗了 {len(cleaned_files)} 个文件")

if __name__ == "__main__":
    main() 
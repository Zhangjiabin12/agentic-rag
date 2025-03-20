import sys
import os
import re
import json
import argparse
from typing import List, Dict
from pathlib import Path
from transformers import AutoTokenizer

from config import config

class TextSplitter:
    """文本分片器，支持多种分片策略"""
    
    def __init__(self, 
                 chunk_size: int = None, 
                 chunk_overlap: int = None):
        """初始化分片器
        
        Args:
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块重叠大小（字符数）一般为chunk_size的10~20%
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    
    def split_by_separator(self, 
                          text: str, 
                          separators: List[str] = None) -> List[str]:
        """按照分隔符分块
        
        Args:
            text: 待分块文本
            separators: 分隔符列表，按优先级排序
            
        Returns:
            分块后的文本列表
        """
        # 使用默认分隔符
        if separators is None:
            separators = config.DEFAULT_SEPARATORS
            
        # 如果文本长度小于分块大小，直接返回
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        # 尝试每个分隔符
        for separator in separators:
            if separator in text:
                # 按分隔符分割文本
                splits = text.split(separator)
                current_chunk = []
                current_length = 0
                
                for split in splits:
                    # 如果当前分块加上新的分割部分不超过分块大小，则添加到当前分块
                    if current_length + len(split) + len(separator) <= self.chunk_size:
                        current_chunk.append(split)
                        current_length += len(split) + len(separator)
                    else:
                        # 如果当前分块不为空，则添加到结果中
                        if current_chunk:
                            chunk_text = separator.join(current_chunk)
                            # 确保每个块不超过chunk_size
                            if len(chunk_text) > self.chunk_size:
                                print(f"警告：生成的块大小({len(chunk_text)})超过了设定的chunk_size({self.chunk_size})，将被截断")
                                chunk_text = chunk_text[:self.chunk_size]
                            chunks.append(chunk_text)
                        
                        # 如果单个分割部分超过分块大小，则需要进一步分割
                        if len(split) > self.chunk_size:
                            # 使用字符分块方法处理过长的部分
                            sub_chunks = self.split_by_chars(split)
                            chunks.extend(sub_chunks)
                            current_chunk = []
                            current_length = 0
                        else:
                            # 开始新的分块
                            current_chunk = [split]
                            current_length = len(split)
                
                # 添加最后一个分块
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    # 确保每个块不超过chunk_size
                    if len(chunk_text) > self.chunk_size:
                        print(f"警告：生成的块大小({len(chunk_text)})超过了设定的chunk_size({self.chunk_size})，将被截断")
                        chunk_text = chunk_text[:self.chunk_size]
                    chunks.append(chunk_text)
                
                # 如果成功分块，则返回结果
                if chunks:
                    return chunks
        
        # 如果所有分隔符都不能成功分块，则按字符分块
        return self.split_by_chars(text)
    
    def split_by_chars(self, text: str) -> List[str]:
        """按字符分块（最基本的分块方法）
        
        Args:
            text: 待分块文本
            
        Returns:
            分块后的文本列表
        """
        chunks = []
        # 确保严格按照chunk_size分块
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            # 确保不超过文本长度
            end = min(i + self.chunk_size, len(text))
            # 添加分块
            chunk = text[i:end]
            # 确保每个块不超过chunk_size
            if len(chunk) > self.chunk_size:
                print(f"警告：生成的块大小({len(chunk)})超过了设定的chunk_size({self.chunk_size})，将被截断")
                chunk = chunk[:self.chunk_size]
            chunks.append(chunk)
            # 如果已经到达文本末尾，则结束
            if end == len(text):
                break
        return chunks
    
    def split_by_tokens(self, 
                       text: str, 
                       tokenizer_name: str = "bert-base-chinese") -> List[str]:
        """按token分块
        
        Args:
            text: 待分块文本
            tokenizer_name: 分词器名称
            
        Returns:
            分块后的文本列表
        """
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 对文本进行分词
        tokens = tokenizer.encode(text)
        
        # 按token分块
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            # 确保不超过token长度
            end = min(i + self.chunk_size, len(tokens))
            # 解码token
            chunk = tokenizer.decode(tokens[i:end])
            
            # 确保每个块的字符数不超过chunk_size
            if len(chunk) > self.chunk_size:
                print(f"警告：生成的块大小({len(chunk)})超过了设定的chunk_size({self.chunk_size})，将被截断")
                # 尝试按字符截断，但保持完整的词
                truncated_chunk = chunk[:self.chunk_size]
                # 尝试在最后一个完整的词或标点处截断
                for sep in config.DEFAULT_SEPARATORS:
                    if sep in truncated_chunk[-20:]:  # 在最后20个字符中查找分隔符
                        last_sep_pos = truncated_chunk.rfind(sep)
                        if last_sep_pos > 0:  # 确保找到了分隔符
                            truncated_chunk = truncated_chunk[:last_sep_pos + len(sep)]
                            break
                chunk = truncated_chunk
            
            # 添加分块
            chunks.append(chunk)
            # 如果已经到达文本末尾，则结束
            if end == len(tokens):
                break
        
        return chunks
    
    def split_by_markdown(self, text: str) -> List[str]:
        """按Markdown标题分块
        
        Args:
            text: 待分块文本
            
        Returns:
            分块后的文本列表
        """
        # 匹配Markdown标题
        pattern = r'^#{1,6}\s+.+$'
        
        # 按行分割文本
        lines = text.split('\n')
        
        # 查找所有标题行的索引
        title_indices = [i for i, line in enumerate(lines) if re.match(pattern, line)]
        
        # 如果没有找到标题，则按分隔符分块
        if not title_indices:
            return self.split_by_separator(text)
        
        # 按标题分块
        chunks = []
        for i in range(len(title_indices)):
            start = title_indices[i]
            end = title_indices[i + 1] if i + 1 < len(title_indices) else len(lines)
            
            # 提取当前块的文本
            chunk = '\n'.join(lines[start:end])
            
            # 如果块太大，则进一步分块
            if len(chunk) > self.chunk_size:
                # 使用分隔符分块方法处理过长的部分
                sub_chunks = self.split_by_separator(chunk)
                chunks.extend(sub_chunks)
            else:
                # 确保每个块不超过chunk_size
                if len(chunk) > self.chunk_size:
                    print(f"警告：生成的块大小({len(chunk)})超过了设定的chunk_size({self.chunk_size})，将被截断")
                    chunk = chunk[:self.chunk_size]
                chunks.append(chunk)
        
        return chunks
    
    def split_file(self, 
                  input_file: str, 
                  output_dir: str = None, 
                  split_method: str = None) -> List[str]:
        """分割文件，确保使用UTF-8编码
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            split_method: 分块方法，支持 separator、chars、tokens、markdown
            
        Returns:
            分块后的文件路径列表
        """
        # 使用默认分块方法和输出目录
        if split_method is None:
            split_method = config.SPLIT_METHOD
        
        if output_dir is None:
            output_dir = config.SPLIT_PATH
            
        # 读取文件（使用UTF-8编码）
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # 如果UTF-8解码失败，尝试读取元数据获取原始编码
            metadata_file = input_file.replace('.txt', '.json')
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        original_encoding = metadata.get('original_encoding')
                        if original_encoding:
                            print(f"文件 {input_file} 使用原始编码 {original_encoding} 读取")
                            with open(input_file, 'r', encoding=original_encoding) as f:
                                text = f.read()
                        else:
                            raise ValueError(f"无法确定文件编码: {input_file}")
                except Exception as e:
                    raise ValueError(f"读取文件失败: {input_file}, 错误: {e}")
            else:
                raise ValueError(f"无法以UTF-8编码读取文件: {input_file}")
        
        # 根据分块方法选择分块函数
        if split_method == 'separator':
            chunks = self.split_by_separator(text)
        elif split_method == 'chars':
            chunks = self.split_by_chars(text)
        elif split_method == 'tokens':
            chunks = self.split_by_tokens(text)
        elif split_method == 'markdown':
            chunks = self.split_by_markdown(text)
        else:
            raise ValueError(f"不支持的分块方法: {split_method}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取文件名（不包含扩展名）
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # 保存分块（使用UTF-8编码）
        output_files = []
        for i, chunk in enumerate(chunks):
            # 构建输出文件路径
            output_file = os.path.join(output_dir, f"{file_name}_chunk_{i+1}.txt")
            
            # 保存分块（使用UTF-8编码）
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
            
            # 保存元数据（使用UTF-8编码）
            metadata_file = os.path.join(output_dir, f"{file_name}_chunk_{i+1}.json")
            
            # 读取原始元数据（如果存在）
            original_metadata = {}
            original_metadata_file = input_file.replace('.txt', '.json')
            if os.path.exists(original_metadata_file):
                try:
                    with open(original_metadata_file, 'r', encoding='utf-8') as f:
                        original_metadata = json.load(f)
                except:
                    pass
            
            # 合并元数据
            metadata = {
                "source": input_file,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "split_method": split_method,
                "encoding": "utf-8"  # 标记使用UTF-8编码
            }
            
            # 保留原始编码信息
            if "original_encoding" in original_metadata:
                metadata["original_encoding"] = original_metadata["original_encoding"]
                metadata["converted_to_utf8"] = True
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            output_files.append(output_file)
        
        return output_files
    
    def split_directory(self, 
                       input_dir: str, 
                       output_dir: str = None, 
                       split_method: str = None, 
                       extensions: List[str] = ['.txt', '.md']) -> Dict[str, List[str]]:
        """分割目录下的所有文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            split_method: 分块方法，支持 separator、chars、tokens、markdown
            extensions: 文件扩展名列表
            
        Returns:
            分块结果字典，键为输入文件路径，值为分块后的文件路径列表
        """
        # 使用默认分块方法和输出目录
        if split_method is None:
            split_method = config.SPLIT_METHOD
        
        if output_dir is None:
            output_dir = config.SPLIT_PATH
            
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储分块结果
        results = {}
        
        # 获取所有文件
        files = []
        for ext in extensions:
            files.extend(list(Path(input_dir).glob(f"**/*{ext}")))
        
        # 分割每个文件
        for file_path in files:
            try:
                # 构建相对路径
                rel_path = os.path.relpath(file_path, input_dir)
                
                # 构建输出目录
                file_output_dir = os.path.join(output_dir, os.path.dirname(rel_path))
                os.makedirs(file_output_dir, exist_ok=True)
                
                # 分割文件
                output_files = self.split_file(str(file_path), file_output_dir, split_method)
                
                # 添加到结果字典
                results[str(file_path)] = output_files
                
                print(f"已分割: {file_path} -> {len(output_files)} 个分块")
            except Exception as e:
                print(f"分割文件失败: {file_path}, 错误: {e}")
        
        # 保存分块信息
        chunks_info_file = os.path.join(output_dir, "chunks_info.json")
        with open(chunks_info_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='文本分片工具')
    parser.add_argument('--input_dir', type=str, help='输入目录')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--chunk_size', type=int, default=config.CHUNK_SIZE, help='分块大小')
    parser.add_argument('--chunk_overlap', type=int, default=config.CHUNK_OVERLAP, help='分块重叠大小')
    parser.add_argument('--split_method', type=str, default=config.SPLIT_METHOD, 
                        choices=['separator', 'chars', 'tokens', 'markdown'], help='分块方法')
    parser.add_argument('--extensions', type=str, nargs='+', default=['.txt', '.md'], help='文件扩展名列表')
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.input_dir:
        parser.error("请指定输入目录")
    
    if not args.output_dir:
        args.output_dir = config.SPLIT_PATH
    
    # 创建分片器
    splitter = TextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    
    # 分割目录
    results = splitter.split_directory(args.input_dir, args.output_dir, args.split_method, args.extensions)
    
    # 统计结果
    total_files = len(results)
    total_chunks = sum(len(chunks) for chunks in results.values())
    print(f"共处理了 {total_files} 个文件，生成了 {total_chunks} 个分块")

if __name__ == "__main__":
    main() 
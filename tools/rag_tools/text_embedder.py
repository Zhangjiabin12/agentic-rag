import os
import json
import hashlib
import argparse
import numpy as np
from typing import List, Dict, Optional, Any
from pathlib import Path
import chromadb
import torch
import time
import sys

from config import config
from tools.rag_tools.vector_retriever import get_embedder_model

class CachedEmbedder:
    """带缓存功能的嵌入器"""
    
    def __init__(self, 
                model_name: str = None,
                cache_dir: str = None):
        """初始化嵌入器
        
        Args:
            model_name: 模型名称
            cache_dir: 缓存目录
        """
        self.model_name = model_name or config.DEFAULT_EMBEDDING_MODEL
        self.cache_dir = cache_dir or config.EMBEDDING_CACHE_PATH
        
        # 使用全局缓存获取模型
        self.model = get_embedder_model(self.model_name)
        
        # 创建模型特定的缓存子目录
        self.model_cache_dir = os.path.join(self.cache_dir, self.model_name.replace('/', '_'))
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # 初始化缓存
        self.cache = self._load_cache()
        
        # 打印设备信息
        print(f"使用设备: {config.DEVICE}")
        if config.USE_GPU:
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """加载缓存"""
        cache_file = os.path.join(self.model_cache_dir, "embedding_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """保存缓存"""
        cache_file = os.path.join(self.model_cache_dir, "embedding_cache.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f)
    
    def _get_cache_key(self, text: str) -> str:
        """获取缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def embed(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """将文本转换为向量
        
        Args:
            texts: 文本列表
            use_cache: 是否使用缓存
            
        Returns:
            向量数组
        """
        if not use_cache:
            return self.model.encode(texts, show_progress_bar=True)
        
        # 使用缓存
        embeddings = []
        texts_to_embed = []
        indices = []
        
        # 检查缓存
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
            else:
                texts_to_embed.append(text)
                indices.append(i)
        
        # 计算未缓存的嵌入
        if texts_to_embed:
            new_embeddings = self.model.encode(texts_to_embed, show_progress_bar=True)
            
            # 更新缓存
            for i, embedding in enumerate(new_embeddings):
                cache_key = self._get_cache_key(texts_to_embed[i])
                self.cache[cache_key] = embedding.tolist()
                embeddings.append(embedding.tolist())
            
            # 保存缓存
            self._save_cache()
        
        # 重新排序嵌入
        sorted_embeddings = [None] * len(texts)
        for i, embedding in enumerate(embeddings):
            if i < len(indices):
                sorted_embeddings[indices[i]] = embedding
            else:
                # 找到第一个空位置
                for j in range(len(sorted_embeddings)):
                    if sorted_embeddings[j] is None:
                        sorted_embeddings[j] = embedding
                        break
        
        return np.array(sorted_embeddings)

class VectorDB:
    """向量数据库"""
    
    def __init__(self, db_path: str = None):
        """初始化向量数据库
        
        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path or config.VECTOR_DB_PATH
        
        # 确保数据库目录存在
        os.makedirs(self.db_path, exist_ok=True)
        
        # 初始化ChromaDB客户端，设置相关参数以支持中文
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=chromadb.Settings(
                anonymized_telemetry=False,  # 禁用遥测
                allow_reset=True,  # 允许重置
                is_persistent=True  # 启用持久化
            )
        )
    
    def create_collection(self, collection_name: str) -> Any:
        """创建集合
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合对象
        """
        try:
            # 尝试获取现有集合
            return self.client.get_collection(name=collection_name)
        except ValueError as e:
            # 如果集合不存在，创建新集合
            try:
                return self.client.create_collection(
                    name=collection_name,
                    metadata={"description": f"Collection for {collection_name}"}
                )
            except Exception as create_error:
                import logging
                logging.error(f"创建集合失败: {collection_name}, 错误: {str(create_error)}")
                raise RuntimeError(f"创建集合失败: {collection_name}") from create_error
    
    def add_documents(self, collection_name: str, documents: List[str], metadatas: List[Dict], embeddings: np.ndarray):
        """添加文档
        
        Args:
            collection_name: 集合名称
            documents: 文档列表
            metadatas: 元数据列表
            embeddings: 向量数组
        """
        # 获取集合
        collection = self.create_collection(collection_name)
        
        # 生成唯一ID，使用更安全的方式处理中文
        ids = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            # 使用索引和文档内容的哈希组合生成唯一ID
            content_hash = hashlib.sha256(f"{doc}{str(meta)}".encode('utf-8')).hexdigest()[:16]
            doc_id = f"doc_{i}_{content_hash}"
            ids.append(doc_id)
        
        try:
            # 添加文档
            collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings.tolist(),
                ids=ids
            )
        except Exception as e:
            import logging
            logging.error(f"添加文档到集合失败: {collection_name}, 错误: {str(e)}")
            raise RuntimeError(f"添加文档到集合失败: {collection_name}") from e
    
    def search(self, collection_name: str, query: str, n_results: int = 5, query_embedding: Optional[List[float]] = None):
        """搜索
        
        Args:
            collection_name: 集合名称
            query: 查询文本
            n_results: 返回结果数量
            query_embedding: 查询向量
            
        Returns:
            搜索结果
        """
        try:
            # 获取集合
            collection = self.create_collection(collection_name)
            
            # 搜索
            if query_embedding is not None:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
            else:
                results = collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
            
            return results
        except Exception as e:
            import logging
            logging.error(f"搜索集合失败: {collection_name}, 错误: {str(e)}")
            raise RuntimeError(f"搜索集合失败: {collection_name}") from e

class TextEmbedder:
    """文本嵌入器，将文本转换为向量并存入向量数据库"""
    
    @staticmethod
    def normalize_collection_name(name: str) -> str:
        """规范化集合名称，使其符合ChromaDB的要求
        
        Args:
            name: 原始集合名称
            
        Returns:
            规范化后的集合名称
        """
        import re
        import pinyin
        
        # 如果名称全是ASCII字符，只需要替换非法字符
        if all(ord(c) < 128 for c in name):
            # 替换非法字符为下划线
            normalized = re.sub(r'[^a-zA-Z0-9-]', '_', name)
        else:
            # 对于包含中文的名称，转换为拼音
            normalized = pinyin.get(name, format='strip', delimiter='_')
            # 替换任何非法字符为下划线
            normalized = re.sub(r'[^a-zA-Z0-9-]', '_', normalized)
        
        # 确保以字母开头（如果以数字开头，添加前缀）
        if normalized[0].isdigit():
            normalized = f"col_{normalized}"
        
        # 如果长度小于3，添加填充
        if len(normalized) < 3:
            normalized = normalized + "_col"
        
        # 如果长度超过63，截断
        if len(normalized) > 63:
            normalized = normalized[:63]
        
        # 确保以字母或数字结尾
        if not normalized[-1].isalnum():
            normalized = normalized + "x"
        
        # 存储原始名称到规范化名称的映射
        if not hasattr(TextEmbedder, '_name_mapping'):
            TextEmbedder._name_mapping = {}
        TextEmbedder._name_mapping[normalized] = name
        
        return normalized
    
    @staticmethod
    def get_original_name(normalized_name: str) -> str:
        """获取规范化名称对应的原始名称
        
        Args:
            normalized_name: 规范化后的名称
            
        Returns:
            原始名称
        """
        if not hasattr(TextEmbedder, '_name_mapping'):
            return normalized_name
        return TextEmbedder._name_mapping.get(normalized_name, normalized_name)
    
    def __init__(self, 
                 model_name: str = None,
                 cache_dir: str = None,
                 db_path: str = None):
        """初始化嵌入器
        
        Args:
            model_name: 模型名称
            cache_dir: 缓存目录
            db_path: 数据库路径
        """
        self.model_name = model_name or config.DEFAULT_EMBEDDING_MODEL
        self.cache_dir = cache_dir or config.EMBEDDING_CACHE_PATH
        self.db_root_path = db_path or config.VECTOR_DB_PATH
        
        # 初始化嵌入器
        self.embedder = CachedEmbedder(model_name=self.model_name, cache_dir=self.cache_dir)
        
        # 确保数据库根目录存在
        os.makedirs(self.db_root_path, exist_ok=True)
        
        # 初始化ChromaDB客户端，设置相关参数以支持中文
        self.client = chromadb.PersistentClient(
            path=self.db_root_path,
            settings=chromadb.Settings(
                anonymized_telemetry=False,  # 禁用遥测
                allow_reset=True,  # 允许重置
                is_persistent=True  # 启用持久化
            )
        )
        
        # 初始化向量数据库
        self.vector_db = VectorDB(db_path=self.db_root_path)
        
        # 初始化名称映射字典
        if not hasattr(TextEmbedder, '_name_mapping'):
            TextEmbedder._name_mapping = {}
            
        # 加载现有集合的名称映射
        try:
            collections = self.client.list_collections()
            for collection in collections:
                if collection.metadata and "original_name" in collection.metadata:
                    TextEmbedder._name_mapping[collection.name] = collection.metadata["original_name"]
        except Exception as e:
            print(f"加载集合名称映射失败: {e}")
    
    def get_db_path_for_collection(self, collection_name: str) -> str:
        """获取集合的数据库路径
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合的数据库路径
        """
        # 创建集合特定的数据库目录
        db_path = os.path.join(self.db_root_path, collection_name)
        os.makedirs(db_path, exist_ok=True)
        return db_path
    
    def embed_file(self, file_path: str, collection_name: str = None) -> None:
        """将文件转换为向量并存入向量数据库
        
        Args:
            file_path: 文件路径
            collection_name: 集合名称
        """
        # 使用默认集合名称
        if collection_name is None:
            collection_name = config.COLLECTION_NAME
            
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 嵌入文本
        embedding = self.embedder.embed([text])[0]
        
        # 获取或创建集合
        try:
            collection = self.client.get_collection(name=collection_name)
        except:
            collection = self.client.create_collection(name=collection_name)
        
        # 添加文档
        collection.add(
            ids=[os.path.basename(file_path)],
            embeddings=[embedding.tolist()],
            documents=[text],
            metadatas=[{"source": os.path.basename(file_path)}]
        )
    
    def embed_directory(self, input_dir: str, collection_name: str = None, extensions: List[str] = ['.txt']) -> None:
        """将目录中的所有文件转换为向量并存入向量数据库
        
        Args:
            input_dir: 输入目录
            collection_name: 集合名称
            extensions: 文件扩展名列表
        """
        # 使用默认集合名称
        if collection_name is None:
            collection_name = config.COLLECTION_NAME
            
        # 获取所有文件
        files = []
        for ext in extensions:
            files.extend(list(Path(input_dir).glob(f"**/*{ext}")))
        
        # 读取所有文件
        texts = []
        ids = []
        metadatas = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                texts.append(text)
                ids.append(os.path.basename(file_path))
                metadatas.append({"source": os.path.basename(file_path)})
            except Exception as e:
                print(f"读取文件失败: {file_path}, 错误: {e}")
        
        # 嵌入文本
        embeddings = self.embedder.embed(texts)
        
        # 获取或创建集合
        try:
            collection = self.client.get_collection(name=collection_name)
        except:
            collection = self.client.create_collection(name=collection_name)
        
        # 添加文档
        collection.add(
            ids=ids,
            embeddings=[embedding.tolist() for embedding in embeddings],
            documents=texts,
            metadatas=metadatas
        )
    
    def embed_chunks_directory(self, input_dir: str, collection_name: str = None, extensions: List[str] = ['.txt']) -> None:
        """将分块目录中的所有文件转换为向量并存入向量数据库
        
        Args:
            input_dir: 输入目录
            collection_name: 集合名称
            extensions: 文件扩展名列表
        """
        # 使用默认集合名称
        if collection_name is None:
            collection_name = config.COLLECTION_NAME
        
        # 规范化集合名称
        original_name = collection_name
        normalized_name = self.normalize_collection_name(collection_name)
        
        print(f"原始集合名称: {original_name}")
        print(f"规范化后的集合名称: {normalized_name}")
        
        # 获取集合特定的数据库路径
        db_path = self.get_db_path_for_collection(normalized_name)
        
        # 获取所有文件
        files = []
        for ext in extensions:
            files.extend(list(Path(input_dir).glob(f"**/*{ext}")))
        
        # 读取所有文件
        texts = []
        ids = []
        metadatas = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # 读取元数据
                metadata_file = file_path.with_suffix('.json')
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                else:
                    metadata = {"source": os.path.basename(file_path)}
                
                # 在元数据中添加原始集合名称
                metadata["collection"] = original_name
                
                # 使用文件路径和内容生成唯一ID
                unique_id = f"{normalized_name}_{hashlib.md5((str(file_path.absolute()) + text[:100]).encode('utf-8')).hexdigest()[:16]}"
                
                texts.append(text)
                ids.append(unique_id)
                metadatas.append(metadata)
            except Exception as e:
                print(f"读取文件失败: {file_path}, 错误: {e}")
        
        if not texts:
            print(f"没有找到任何文件需要处理")
            return
        
        # 嵌入文本
        embeddings = self.embedder.embed(texts)
        
        # 获取或创建集合
        try:
            # 使用集合特定的客户端
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_or_create_collection(
                name=normalized_name,
                metadata={"original_name": original_name}  # 在集合元数据中存储原始名称
            )
            
            # 添加文档
            collection.add(
                ids=ids,
                embeddings=[embedding.tolist() for embedding in embeddings],
                documents=texts,
                metadatas=metadatas
            )
            
            print(f"成功将 {len(texts)} 个文档嵌入到集合 {original_name} 中")
            print(f"数据库路径: {db_path}")
            print(f"规范化名称: {normalized_name}")
            
        except Exception as e:
            print(f"处理集合失败: {original_name}, 错误: {str(e)}")
            raise
    
    def embed_text(self, text: str, collection_name: str = None, unique_id: str = None, metadata: Dict = None) -> None:
        """将文本转换为向量并存入向量数据库
        
        Args:
            text: 文本内容
            collection_name: 集合名称
            unique_id: 唯一ID，如果为None则自动生成
            metadata: 元数据，如果为None则使用空字典
        """
        # 使用默认集合名称
        if collection_name is None:
            collection_name = config.COLLECTION_NAME
        
        # 获取集合特定的数据库路径
        db_path = self.get_db_path_for_collection(collection_name)
        
        # 使用默认元数据
        if metadata is None:
            metadata = {}
        
        # 生成唯一ID
        if unique_id is None:
            unique_id = f"{collection_name}_{hashlib.md5(text[:100].encode('utf-8')).hexdigest()[:16]}"
        
        # 嵌入文本
        embedding = self.embedder.embed([text])[0]
        
        # 获取或创建集合
        try:
            # 使用集合特定的客户端
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_or_create_collection(name=collection_name)
        except Exception as e:
            print(f"创建集合失败: {collection_name}, 错误: {e}")
            return
        
        # 添加文档
        collection.add(
            ids=[unique_id],
            embeddings=[embedding.tolist()],
            documents=[text],
            metadatas=[metadata]
        )
        
        print(f"成功将文本嵌入到集合 {collection_name} 中，数据库路径: {db_path}")

    def _check_embedding_dimension(self, collection_name: str, embedding: np.ndarray) -> bool:
        """检查向量维度是否与集合中的其他向量一致
        
        Args:
            collection_name: 集合名称
            embedding: 待检查的向量
            
        Returns:
            是否一致
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            existing_embeddings = collection.get()
            if existing_embeddings['embeddings']:
                existing_dim = len(existing_embeddings['embeddings'][0])
                current_dim = len(embedding)
                if existing_dim != current_dim:
                    print(f"维度不匹配: 集合维度 {existing_dim}, 当前维度 {current_dim}")
                    return False
        except ValueError:
            # 集合不存在，无需检查
            pass
        return True
    
    def _get_model_dimension(self) -> int:
        """获取当前模型的输出维度"""
        # 使用一个简单的文本测试模型输出维度
        test_embedding = self.embedder.embed(["测试文本"])[0]
        return len(test_embedding)
    
    def add_documents(self, collection_name: str, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None) -> None:
        """添加文档到向量数据库
        
        Args:
            collection_name: 集合名称
            documents: 文档列表
            metadatas: 元数据列表
            ids: ID列表
        """
        # 获取集合特定的数据库路径
        db_path = os.path.join(self.db_root_path, collection_name)
        os.makedirs(db_path, exist_ok=True)
        
        # 使用集合特定的客户端
        client = chromadb.PersistentClient(path=db_path)
        
        try:
            # 尝试获取现有集合
            collection = client.get_collection(name=collection_name)
        except Exception:
            # 如果集合不存在，创建新集合
            collection = client.create_collection(
                name=collection_name,
                metadata={"embedding_model": self.model_name}
            )
        
        # 生成嵌入向量
        embeddings = self.embedder.embed(documents)
        
        # 添加到集合
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"成功将 {len(documents)} 个文档添加到集合 {collection_name} 中")
        print(f"数据库路径: {db_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='文本嵌入工具')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录')
    parser.add_argument('--collection_name', type=str, help='集合名称')
    parser.add_argument('--model_name', type=str, help='模型名称')
    parser.add_argument('--cache_dir', type=str, help='缓存目录')
    parser.add_argument('--db_path', type=str, help='数据库路径')
    parser.add_argument('--is_chunks', action='store_true', help='是否为分块目录')
    parser.add_argument('--extensions', type=str, nargs='+', default=['.txt'], help='文件扩展名列表')
    
    args = parser.parse_args()
    
    # 创建嵌入器
    embedder = TextEmbedder(
        model_name=args.model_name or config.DEFAULT_EMBEDDING_MODEL,
        cache_dir=args.cache_dir or config.EMBEDDING_CACHE_PATH,
        db_path=args.db_path or config.VECTOR_DB_PATH
    )
    
    # 嵌入文本
    if args.is_chunks:
        embedder.embed_chunks_directory(args.input_dir, args.collection_name or config.COLLECTION_NAME, args.extensions)
    else:
        embedder.embed_directory(args.input_dir, args.collection_name or config.COLLECTION_NAME, args.extensions)

if __name__ == "__main__":
    main() 
import os
import argparse
import numpy as np
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import sys
from pathlib import Path

from config import config

# 全局模型缓存
_GLOBAL_EMBEDDER_MODELS = {}
_GLOBAL_RERANKER_MODELS = {}

def get_embedder_model(model_name: str):
    """获取嵌入模型，如果已经加载则直接返回，否则加载模型
    
    Args:
        model_name: 模型名称
        
    Returns:
        加载的模型
    """
    global _GLOBAL_EMBEDDER_MODELS
    if model_name not in _GLOBAL_EMBEDDER_MODELS:
        print(f"首次加载嵌入模型: {model_name}，后续将从缓存中获取")
        _GLOBAL_EMBEDDER_MODELS[model_name] = SentenceTransformer(model_name, device=config.DEVICE)
    return _GLOBAL_EMBEDDER_MODELS[model_name]

def get_reranker_model(model_name: str):
    """获取重排序模型，如果已经加载则直接返回，否则加载模型
    
    Args:
        model_name: 模型名称
        
    Returns:
        加载的模型
    """
    global _GLOBAL_RERANKER_MODELS
    if model_name not in _GLOBAL_RERANKER_MODELS:
        print(f"首次加载重排序模型: {model_name}，后续将从缓存中获取")
        _GLOBAL_RERANKER_MODELS[model_name] = CrossEncoder(model_name)
    return _GLOBAL_RERANKER_MODELS[model_name]

class Embedder:
    """嵌入器"""
    
    def __init__(self, model_name: str = None):
        """初始化嵌入器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name or config.DEFAULT_EMBEDDING_MODEL
        # 使用全局缓存获取模型
        self.model = get_embedder_model(self.model_name)
    
    def embed(self, text: str) -> np.ndarray:
        """将文本转换为向量
        
        Args:
            text: 文本
            
        Returns:
            向量
        """
        return self.model.encode(text)

class Reranker:
    """重排序器"""
    
    def __init__(self, model_name: str = None):
        """初始化重排序器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name or config.DEFAULT_RERANK_MODEL
        # 使用全局缓存获取模型
        self.model = get_reranker_model(self.model_name)
    
    def rerank(self, query: str, documents: List[str]) -> List[Dict]:
        """重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            重排序后的文档列表，包含分数
        """
        # 构建查询-文档对
        query_doc_pairs = [[query, doc] for doc in documents]
        
        # 计算分数
        scores = self.model.predict(query_doc_pairs)
        
        # 构建结果
        results = [{"document": doc, "score": float(score)} for doc, score in zip(documents, scores)]
        
        # 按分数降序排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results

class VectorDB:
    """向量数据库"""
    
    def __init__(self, db_path: str = None):
        """初始化向量数据库
        
        Args:
            db_path: 数据库路径
        """
        self.db_root_path = db_path or config.VECTOR_DB_PATH
        
        # 确保数据库根目录存在
        if not os.path.exists(self.db_root_path):
            raise ValueError(f"数据库根路径不存在: {self.db_root_path}")
    
    def get_db_path_for_collection(self, collection_name: str) -> str:
        """获取集合的数据库路径
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合的数据库路径
        """
        # 创建集合特定的数据库目录
        db_path = os.path.join(self.db_root_path, collection_name)
        
        # 检查路径是否存在
        if not os.path.exists(db_path):
            raise ValueError(f"集合的数据库路径不存在: {db_path}")
            
        return db_path
    
    def get_collection(self, collection_name: str) -> Any:
        """获取集合
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合对象
        """
        try:
            # 获取集合特定的数据库路径
            db_path = self.get_db_path_for_collection(collection_name)
            
            # 使用集合特定的客户端
            client = chromadb.PersistentClient(path=db_path)
            return client.get_collection(name=collection_name)
        except Exception as e:
            raise ValueError(f"获取集合失败: {collection_name}, 错误: {e}")
    
    def search(self, collection_name: str, query: str, n_results: int = None, query_embedding: Optional[np.ndarray] = None):
        """搜索
        
        Args:
            collection_name: 集合名称
            query: 查询文本
            n_results: 返回结果数量
            query_embedding: 查询向量
            
        Returns:
            搜索结果
        """
        # 使用默认返回结果数量
        if n_results is None:
            n_results = config.TOP_K
            
        # 获取集合
        collection = self.get_collection(collection_name)
        
        # 搜索
        if query_embedding is not None:
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
        else:
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
        
        return results
    
    def list_collections(self) -> List[str]:
        """列出所有集合
        
        Returns:
            集合名称列表
        """
        # 获取所有子目录作为集合名称
        collections = []
        for item in os.listdir(self.db_root_path):
            item_path = os.path.join(self.db_root_path, item)
            if os.path.isdir(item_path):
                try:
                    # 检查是否是有效的集合目录
                    client = chromadb.PersistentClient(path=item_path)
                    colls = client.list_collections()
                    if item in colls:
                        collections.append(item)
                except:
                    # 如果不是有效的集合目录，则跳过
                    pass
        
        return collections

class VectorRetriever:
    """向量检索器"""
    
    def __init__(self, 
                 db_path: str = None,
                 embedder_model: str = None,
                 reranker_model: str = None,
                 use_reranker: bool = None):
        """初始化向量检索器
        
        Args:
            db_path: 数据库根路径
            embedder_model: 嵌入模型名称
            reranker_model: 重排序模型名称
            use_reranker: 是否使用重排序
        """
        self.db_root_path = db_path or config.VECTOR_DB_PATH
        self.vector_db = VectorDB(self.db_root_path)
        self.embedder = Embedder(embedder_model or config.DEFAULT_EMBEDDING_MODEL)
        self.use_reranker = use_reranker if use_reranker is not None else config.USE_RERANKER
        if self.use_reranker:
            self.reranker = Reranker(reranker_model or config.DEFAULT_RERANK_MODEL)
    
    def retrieve(self, query: str, collection_name: str = None, top_k: int = None) -> Dict:
        """检索
        
        Args:
            query: 查询文本
            collection_name: 集合名称
            top_k: 返回结果数量
            
        Returns:
            检索结果
        """
        # 使用默认集合名称和返回结果数量
        if collection_name is None:
            collection_name = config.COLLECTION_NAME
        
        if top_k is None:
            top_k = config.TOP_K
            
        # 嵌入查询
        query_embedding = self.embedder.embed(query)
        
        # 检索
        try:
            results = self.vector_db.search(collection_name, query, top_k, query_embedding)
            
            # 提取结果
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            # 如果使用重排序，则重排序
            if self.use_reranker and documents:
                reranked_results = self.reranker.rerank(query, documents)
                
                # 重新排序文档和元数据
                reranked_documents = [result["document"] for result in reranked_results]
                reranked_metadatas = []
                reranked_scores = []
                
                # 保持元数据和文档的对应关系
                for doc in reranked_documents:
                    idx = documents.index(doc)
                    reranked_metadatas.append(metadatas[idx])
                    reranked_scores.append(reranked_results[documents.index(doc)]["score"])
                
                return {
                    "query": query,
                    "documents": reranked_documents,
                    "metadatas": reranked_metadatas,
                    "scores": reranked_scores
                }
            
            return {
                "query": query,
                "documents": documents,
                "metadatas": metadatas,
                "scores": distances
            }
        except Exception as e:
            print(f"检索失败: {collection_name}, 错误: {e}")
            return {
                "query": query,
                "documents": [],
                "metadatas": [],
                "scores": []
            }
    
    def list_collections(self) -> List[str]:
        """列出所有集合
        
        Returns:
            集合名称列表
        """
        return self.vector_db.list_collections()

def format_results(results: Dict) -> str:
    """格式化结果
    
    Args:
        results: 检索结果
        
    Returns:
        格式化后的结果
    """
    output = f"查询: {results['query']}\n\n"
    
    for i, (doc, meta, score) in enumerate(zip(results["documents"], results["metadatas"], results["scores"])):
        output += f"结果 {i+1} (分数: {score:.4f}):\n"
        output += f"来源: {meta.get('source', '未知')}\n"
        output += f"文件名: {meta.get('filename', '未知')}\n"
        
        # 如果文本太长，则截断
        if len(doc) > 500:
            doc = doc[:500] + "..."
        
        output += f"内容: {doc}\n\n"
    
    return output

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='向量检索工具')
    parser.add_argument('--db_path', type=str, help='数据库根路径')
    parser.add_argument('--collection_name', type=str, help='集合名称')
    parser.add_argument('--query', type=str, help='查询文本')
    parser.add_argument('--top_k', type=int, help='返回结果数量')
    parser.add_argument('--embedder_model', type=str, help='嵌入模型名称')
    parser.add_argument('--reranker_model', type=str, help='重排序模型名称')
    parser.add_argument('--reranker', action='store_true', help='使用重排序')
    parser.add_argument('--list_collections', action='store_true', help='列出所有集合')
    parser.add_argument('--output_file', type=str, help='输出文件路径')
    
    args = parser.parse_args()
    
    # 创建向量检索器
    retriever = VectorRetriever(
        db_path=args.db_path or config.VECTOR_DB_PATH,
        embedder_model=args.embedder_model or config.DEFAULT_EMBEDDING_MODEL,
        reranker_model=args.reranker_model or config.DEFAULT_RERANK_MODEL,
        use_reranker=args.reranker if args.reranker else config.USE_RERANKER
    )
    
    # 列出所有集合
    if args.list_collections:
        collections = retriever.list_collections()
        if collections:
            print(f"可用集合: {collections}")
        else:
            print("数据库中没有集合")
        return
    
    # 检查参数
    if not args.collection_name:
        collections = retriever.list_collections()
        if collections:
            print(f"请指定集合名称，可用集合: {collections}")
        else:
            print("数据库中没有集合，请先创建集合")
        return
    
    if not args.query:
        print("请指定查询文本")
        return
    
    # 检索
    results = retriever.retrieve(args.query, args.collection_name, args.top_k)
    
    # 检查结果是否为空
    if not results["documents"]:
        print(f"未找到与查询 '{args.query}' 相关的结果，请检查集合 '{args.collection_name}' 是否存在或包含相关文档")
        return
    
    # 格式化结果
    formatted_results = format_results(results)
    
    # 输出结果
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as file:
            file.write(formatted_results)
        print(f"结果已保存到: {args.output_file}")
    else:
        print(formatted_results)

if __name__ == "__main__":
    main() 
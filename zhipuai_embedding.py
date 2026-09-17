from typing import List
from langchain_core.embeddings import Embeddings

class ZhipuAIEmbeddings(Embeddings):
    """`Zhipuai Embeddings` embedding models."""
    def __init__(self):
        """
        实例化ZhipuAI为values["client"]

        Args:

            values (Dict): 包含配置信息的字典，必须包含 client 的字段.
        Returns:

            values (Dict): 包含配置信息的字典。如果环境中有zhipuai库，则将返回实例化的ZhipuAI类；否则将报错 'ModuleNotFoundError: No module named 'zhipuai''.
        """
        from zhipuai import ZhipuAI
        self.client = ZhipuAI()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
    生成输入文本列表的 embedding.
    Args:
        texts (List[str]): 要生成 embedding 的文本列表.

    Returns:
        List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        texts = [t for t in texts if t.strip()]
        print(f"待向量化文本总数：{len(texts)}")
        result = []
        for i in range(0, len(texts), 64):
            batch = texts[i:i+64]
            print(f"正在处理批次 {i} ~ {i+64}, 批次数量：{len(batch)}")
            embeddings = self.client.embeddings.create(
                model="embedding-2",
                input=batch
            )
            result.extend([emb.embedding for emb in embeddings.data])
        return result



    
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """

        return self.embed_documents([text])[0]
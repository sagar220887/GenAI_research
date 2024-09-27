'''
Hybrid Search = Keyword Search + Symantic Search

Keyword Search -> will use BM25 Algorithms -> Sparse Matrix is produced.
Its an extension of the TF-IDF 

Symantic Search -> will use FAISS -> Dense Matrix is produced.

The Hybrid Search will combine the results of both Keyword Search and Symantic Search.
'''
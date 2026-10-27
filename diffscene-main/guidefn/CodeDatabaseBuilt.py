import ast
import json
import os
import ast
import os
from typing import List, Dict, Any, Set
from dataclasses import dataclass

from scipy.constants import precision


@dataclass
class CodeChunk:
    chunk_id: str
    code: str
    chunk_type: str
    name: str
    start_line: int
    end_line: int
    file_path: str
    dependencies: List[str]
    metadata: Dict[str, Any]


class ASTCodeSplitter:
    """基于AST的语义代码分块"""

    def __init__(self, max_chunk_size=100):
        self.max_chunk_size = max_chunk_size
        self.imports_cache = {}

    def split_file(self, file_path) :
        """Split python file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            chunks = []

            parent_map = self._build_parent_map(tree)

            # Extract import
            import_chunk = self._extract_imports(tree, lines, file_path)
            if import_chunk:
                chunks.append(import_chunk)

            # Extract class and method
            chunks.extend(self._extract_classes_and_methods(tree, lines, file_path))

            # Extract function
            chunks.extend(self._extract_functions(tree, lines, file_path, parent_map))

            # Extract global variables
            chunks.extend(self._extract_global_variables(tree, lines, file_path, parent_map))

            return chunks

        except SyntaxError as e:
            print(f"SyntaxError: {file_path} - {e}")
            return []

    def _build_parent_map(self, tree: ast.AST) -> Dict[ast.AST, ast.AST]:
        """Build parent-child relationship mapping for AST nodes"""
        parent_map = {}

        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent

        return parent_map

    def _extract_imports(self, tree, lines, file_path):
        """Extract imports when existing or return None"""
        import_lines = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                import_lines.extend(lines[start_line:end_line + 1])

        if import_lines:
            return CodeChunk(
                chunk_id=f"{file_path}_imports",
                code='\n'.join(import_lines),
                chunk_type="imports",
                name="imports",
                start_line=0,
                end_line=len(import_lines) - 1,
                file_path=file_path,
                dependencies=[],
                metadata={"import_count": len(import_lines)}
            )
        return None

    def _extract_classes_and_methods(self, tree, lines, file_path):
        chunks = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # classes
                class_chunk = self._create_class_chunk(node, lines, file_path)
                chunks.append(class_chunk)

                # methods of class
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        method_chunk = self._create_method_chunk(child, lines, file_path, node.name)
                        chunks.append(method_chunk)

        return chunks

    def _extract_functions(self, tree, lines, file_path, parent_map):
        chunks = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if it is a class method (already handled within the class)
                if not self._is_method(node, parent_map):
                    func_chunk = self._create_function_chunk(node, lines, file_path)
                    chunks.append(func_chunk)

        return chunks

    def _extract_global_variables(self, tree, lines, file_path, parent_map):
        chunks = []
        global_vars = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Check if it is a global scope assignment
                if self._is_global_scope(node, parent_map):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    var_code = '\n'.join(lines[start_line:end_line + 1])

                    # Identify constants
                    var_names = self._get_assigned_names(node)
                    is_constant = all(name.isupper() for name in var_names)

                    chunk = CodeChunk(
                        chunk_id=f"{file_path}_var_{start_line}",
                        code=var_code,
                        chunk_type="constant" if is_constant else "variable",
                        name=f"global_{'_'.join(var_names)}",
                        start_line=start_line,
                        end_line=end_line,
                        file_path=file_path,
                        dependencies=[],
                        metadata={
                            "variables": var_names,
                            "is_constant": is_constant
                        }
                    )
                    chunks.append(chunk)

        return chunks

    def _create_class_chunk(self, node, lines, file_path):
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line

        code = '\n'.join(lines[start_line:end_line])
        dependencies = self._extract_dependencies(node)

        return CodeChunk(
            chunk_id=f"{file_path}_class_{node.name}",
            code=code,
            chunk_type="class",
            name=node.name,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            dependencies=dependencies,
            metadata={
                "bases": [self._get_base_name(base) for base in node.bases],
                "decorators": [self._get_decorator_name(decorator) for decorator in node.decorator_list]
            }
        )

    def _create_method_chunk(self, child, lines, file_path, class_name):
        start_line = child.lineno-1
        end_line = child.end_lineno if hasattr(child, 'end_lineno') else start_line

        code = '\n'.join(lines[start_line:end_line])
        dependencies = self._extract_dependencies(child)

        return CodeChunk(
                chunk_id=f"{file_path}_class_{class_name}_method_{child.name}",
                code=code,
                chunk_type="method",
                name=child.name,
                start_line=start_line,
                end_line=end_line,
                file_path=file_path,
                dependencies=dependencies,
                metadata={
                    "associated_class": class_name
                }
            )

    def _create_function_chunk(self, node, lines, file_path):
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line

        code = '\n'.join(lines[start_line:end_line])
        dependencies = self._extract_dependencies(node)

        return CodeChunk(
            chunk_id=f"{file_path}_function_{node.name}",
            code=code,
            chunk_type="function",
            name=node.name,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            dependencies=dependencies,
            metadata={}
        )


    def _extract_dependencies(self, node):
        dependencies = set()

        for child in ast.walk(node):
            # Extract function calls
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.add(child.func.attr)

            # Extract variable references
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                dependencies.add(child.id)

        return list(dependencies)

    def _is_method(self, node: ast.FunctionDef, parent_map) -> bool:
        parent = parent_map.get(node)
        return isinstance(parent, ast.ClassDef)


    def _is_global_scope(self, node, parent_map):
        current = node
        while current in parent_map:
            parent = parent_map[current]
            if isinstance(parent, (ast.FunctionDef, ast.ClassDef)):
                return False
            current = parent
        return True

    def _get_assigned_names(self, node):
        names = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Tuple):
                names.extend(el.id for el in target.elts if isinstance(el, ast.Name))
        return names

    def _get_base_name(self, base):
        if isinstance(base, ast.Name):
            return base.id
        return "unknown"

    def _get_decorator_name(self, decorator):
        """Obtain the name of the decorator"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        return "unknown"




from sentence_transformers import SentenceTransformer
import numpy as np

class CodeEmbeddingGenerator:

    def __init__(self, model_name='/home/ztj/learn/LD-Scene/model_ckpt/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def generate_embedding(self, text) -> np.ndarray:
        """Encode text converted from code"""
        embedding = self.model.encode(text)
        return embedding.astype(np.float64)

    def generate_code_embedding(self, chunk):
        """Convert code Chunk to text and encode it"""
        embedding_text = self._prepare_embedding_text(chunk)
        return self.generate_embedding(embedding_text)

    def _prepare_embedding_text(self, chunk):
        """Convert code Chunk to text"""
        parts = []

        # Basic information
        parts.append(f"code_type: {chunk.chunk_type}")
        parts.append(f"code_name: {chunk.name}")

        # Detailed code content
        parts.append(f"code_content: {chunk.code}")

        # Dependencies
        if chunk.dependencies:
            parts.append(f"code_dependencies: {', '.join(chunk.dependencies)}")

        # Metadata
        if chunk.metadata:
            metadata_str = self._format_metadata(chunk.metadata)
            parts.append(f"code_metadata: {metadata_str}")

        return "\n".join(parts)

    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata"""
        formatted = []
        for key, value in metadata.items():
            if isinstance(value, list):
                formatted.append(f"{key}: {', '.join(map(str, value))}")
            elif isinstance(value, dict):
                formatted.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
            else:
                formatted.append(f"{key}: {value}")
        return '; '.join(formatted)

import chromadb
from chromadb.config import Settings

class VectorDatabase:
    """Vector Database Manager"""

    def __init__(self, persist_directory="./rag_utils"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.embedding_generator = CodeEmbeddingGenerator()

        # Create collection
        self.collection = self.client.get_or_create_collection(
            name="code_chunks",
            metadata={"description": "store code chunk in database"},
        )

    def store_chunks(self, chunks, batch_size=100):
        """Store code chunks in a vector database"""
        if not chunks:
            print("No code blocks need to be stored")
            return

        # Batch processing to avoid memory issues
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            self._store_batch(batch_chunks)
            print(f"Have processed {min(i + batch_size, len(chunks))}/{len(chunks)} code chunks")

        print("All code chunks are stored!")

    def _store_batch(self, chunks):
        """Bulk storage of code chunks"""
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in chunks:
            # Generate embedding vector
            embedding = self.embedding_generator.generate_code_embedding(chunk)

            # Prepare stored data
            ids.append(chunk.chunk_id)
            documents.append(self._create_document(chunk))
            embeddings.append(embedding.tolist())
            metadatas.append(self._create_metadata(chunk))

        # store chunks into database
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def _create_document(self, chunk):
        doc_parts = [
            f"code_id: {chunk.chunk_id}",
            f"code_type: {chunk.chunk_type}",
            f"code_name: {chunk.name}",
        ]

        if chunk.dependencies:
            doc_parts.append(f"code_dependencies: {', '.join(chunk.dependencies)}")

        if chunk.metadata:
            metadata_info = []
            for key, value in chunk.metadata.items():
                if key in ['description', 'purpose', 'functionality']:
                    metadata_info.append(f"{key}: {value}")
                elif key == 'args' and isinstance(value, list):
                    metadata_info.append(f"args: {', '.join(value)}")
                elif key == 'return_type':
                    metadata_info.append(f"return_type: {value}")

            if metadata_info:
                doc_parts.append(" | ".join(metadata_info))

        doc_parts.append(f"\ncode_content:\n{chunk.code}")

        return "\n".join(doc_parts)

    def _create_metadata(self, chunk):
        metadata = {
            'chunk_type': chunk.chunk_type,
            'name': chunk.name,
            'file_path': chunk.file_path,
            'start_line': chunk.start_line,
            'end_line': chunk.end_line,
            'dependencies': json.dumps(chunk.dependencies, ensure_ascii=False),
        }


        if chunk.metadata:
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                elif isinstance(value, (list, dict)):
                    metadata[key] = json.dumps(value, ensure_ascii=False)
                else:
                    metadata[key] = str(value)

        return metadata

if __name__ == '__main__':
    A = ASTCodeSplitter(100)
    chunks = A.split_file("/home/ztj/learn/LD-Scene/guidefn/helperfunction.py")
    VectorDB = VectorDatabase("./rag_utils")
    VectorDB.store_chunks(chunks)

    # VectorDB= VectorDatabase()
    # query_embeddings=VectorDB.embedding_generator.generate_embedding("Compute collision loss between background vehicles")
    # results = VectorDB.collection.query(
    #     query_embeddings=[query_embeddings],  # 输入查询文本
    #     n_results=3
    # )
    #
    # # 打印结果
    # print("相似文档:")
    # for i, doc in enumerate(results['documents'][0]):
    #     print(f"{i+1}. {doc}")


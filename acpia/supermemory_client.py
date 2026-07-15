import os
import time

import requests


class SupermemoryClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or os.environ["SUPERMEMORY_BASE_URL"]).rstrip("/")
        self.api_key = api_key or os.environ["SUPERMEMORY_API_KEY"]
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {self.api_key}"

    def ingest(self, content: str, container_tag: str) -> str:
        resp = self._session.post(
            f"{self.base_url}/v3/documents",
            json={"content": content, "containerTag": container_tag},
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def get_document(self, doc_id: str) -> dict:
        resp = self._session.get(f"{self.base_url}/v3/documents/{doc_id}")
        resp.raise_for_status()
        return resp.json()

    def wait_for_document(self, doc_id: str, timeout: float = 60.0, poll_interval: float = 2.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            doc = self.get_document(doc_id)
            if doc["status"] in ("done", "failed"):
                return doc
            time.sleep(poll_interval)
        raise TimeoutError(f"document {doc_id} did not finish processing within {timeout}s")

    def list_documents(self, container_tag: str, page_size: int = 100) -> list[dict]:
        """Return every document record for a case (paginated internally)."""
        docs = []
        page = 1
        while True:
            resp = self._session.post(
                f"{self.base_url}/v3/documents/list",
                json={"containerTags": [container_tag], "limit": page_size, "page": page},
            )
            resp.raise_for_status()
            data = resp.json()
            docs.extend(data["memories"])  # API names the document list "memories"
            if page >= data["pagination"]["totalPages"]:
                break
            page += 1
        return docs

    def fetch_all_memories(self, container_tag: str) -> list[dict]:
        """Fetch EVERY extracted memory for a case, each enriched with its source
        document's id, title and createdAt. No search, no ranking — completeness is
        the whole point for timelines/summaries."""
        memories = []
        for doc in self.list_documents(container_tag):
            if doc.get("status") != "done":
                continue
            full = self.get_document(doc["id"])
            for m in full.get("memories", []):
                memories.append(
                    {
                        **m,
                        "source_doc_id": doc["id"],
                        "source_title": doc.get("title"),
                        "source_created_at": doc.get("createdAt"),
                    }
                )
        return memories

    def search(
        self, query: str, container_tag: str, limit: int = 20, threshold: float = 0.3
    ) -> list[dict]:
        # threshold defaults low: in an investigative context, silently dropping a
        # borderline-relevant lead is worse than surfacing noise for a human to dismiss.
        resp = self._session.post(
            f"{self.base_url}/v4/search",
            json={
                "q": query,
                "containerTag": container_tag,
                "limit": limit,
                "threshold": threshold,
            },
        )
        resp.raise_for_status()
        return resp.json()["results"]

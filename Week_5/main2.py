from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()


# A simple GET route
@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}

# A route with a path parameter
@app.get("/items/{item_id}")
def read_item(item_id: int):
    # Notice the type hint `int`? FastAPI will automatically throw a 422 Error 
    # if someone tries to pass a string like "/items/apple"
    return {"item_id": item_id, "status": "Found"}



# Define how your incoming data should look
class Document(BaseModel):
    title: str
    content: str
    tags: list[str] = []         # Optional, defaults to empty list
    author_id: Optional[int] = None # Optional field

@app.post("/documents/")
def create_document(doc: Document):
    # FastAPI has already validated the payload by the time this code runs.
    # If the user forgot 'title', FastAPI automatically returned a 422 error to them.
    
    # You can access fields using dot notation
    print(f"Processing document: {doc.title}")
    
    # Return the data (FastAPI automatically converts it to JSON)
    return {"status": "success", "data": doc}



from fastapi import FastAPI

app = FastAPI()

tasks = [
    {"id": 1, "title": "Learn Python", "completed": True},
    {"id": 2, "title": "Learn FastAPI", "completed": False},
    {"id": 3, "title": "Build a REST API", "completed": False}
]


@app.get("/tasks/{task_id}")
def get_task(task_id: int):
    for task in tasks:
        if task["id"] == task_id:
            return task

    return {"message": "Task not found"}
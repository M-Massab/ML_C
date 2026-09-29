from fastapi import FastAPI

app=FastAPI()

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
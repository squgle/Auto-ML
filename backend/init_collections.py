from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["ai_project_db"]

# USERS collection
users_collection = db["users"]
users_collection.insert_one({
    "name": "Test User",
    "email": "test@example.com",
    "password_hash": "hashed_password_here"
})

# PROMPTS collection
prompts_collection = db["prompts"]
prompts_collection.insert_one({
    "user_id": "user_id_string",
    "prompt_text": "Generate insights from sales data",
    "timestamp": "2025-07-28T10:00:00"
})

# REPORTS collection
reports_collection = db["reports"]
reports_collection.insert_one({
    "user_id": "user_id_string",
    "report_path": "/reports/report_01.pdf",
    "summary": "This report contains sales trend insights",
    "created_at": "2025-07-28T10:05:00"
})

print("Collections and dummy data initialized.")

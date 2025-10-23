# Database configuration and connection
import os
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional

# MongoDB Connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://saravanan:saran1234@cluster0.1d6gi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
client: Optional[AsyncIOMotorClient] = None

async def connect_to_mongo():
    """Connect to MongoDB"""
    global client
    try:
        client = AsyncIOMotorClient(MONGODB_URL)
        # Verify connection
        await client.admin.command('ping')
        print("✅ Connected to MongoDB successfully!")
    except Exception as e:
        print(f"❌ Could not connect to MongoDB: {e}")
        client = None

async def close_mongo_connection():
    """Close MongoDB connection"""
    global client
    if client:
        client.close()
        print("❌ Closed MongoDB connection")

def get_database():
    """Get database instance"""
    if client is not None:
        return client.codeguard_db
    return None

# Collections
def get_users_collection():
    db = get_database()
    return db.users if db is not None else None

def get_submissions_collection():
    db = get_database()
    return db.submissions if db is not None else None

def get_reports_collection():
    db = get_database()
    return db.reports if db is not None else None

def get_statistics_collection():
    db = get_database()
    return db.statistics if db is not None else None

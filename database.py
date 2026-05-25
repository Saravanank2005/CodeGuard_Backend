import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional

# MongoDB Connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://saravanan:amada1234@cluster0.1d6gi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
client: Optional[AsyncIOMotorClient] = None

async def connect_to_mongo():
    """Connect to MongoDB"""
    global client
    try:
        # Set connection timeout to 5 seconds
        client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        # Verify connection with timeout
        await asyncio.wait_for(client.admin.command('ping'), timeout=5.0)
        print("[OK] Connected to MongoDB successfully!")
    except asyncio.TimeoutError:
        print("[WARN] MongoDB connection timed out (5s). Check your internet and MongoDB URI.")
        client = None
    except Exception as e:
        print(f"[ERROR] Could not connect to MongoDB: {e}")
        client = None

async def close_mongo_connection():
    """Close MongoDB connection"""
    global client
    if client:
        client.close()
        print("[INFO] Closed MongoDB connection")

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

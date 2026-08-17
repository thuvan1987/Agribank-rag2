import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Database Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class Roles:
    # Defined Roles for RBAC
    ADMIN = "Admin"
    STAFF = "Staff"
    GUEST = "Guest"

    @classmethod
    def get_valid_roles(cls):
        """Returns a list of all valid roles."""
        return [cls.ADMIN, cls.STAFF, cls.GUEST]

    @classmethod
    def is_valid_role(cls, role):
        """Checks if a given role is valid."""
        return role in cls.get_valid_roles()

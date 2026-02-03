import uvicorn
from utils.config import config

if __name__ == "__main__":
    uvicorn.run(
        "admin.app:app",
        host="0.0.0.0",
        port=config.admin_port,
        reload=False,
    )

import uvicorn

if __name__ == '__main__':
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=8000,
                reload=False,
                ssl_keyfile="./privkey9.pem", 
                ssl_certfile="./cert9.pem"
                )

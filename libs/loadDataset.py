# Reading from file
print("Loading Dataset...")
fileBuffer = open("./datasets/ri.txt", "r", encoding = "utf-8")
file = fileBuffer.read()
fileBuffer.close()

class Man:
	def __init__(self,name):
		self.name = name
		print("Initilized!")
	def hello(self):
		print("Hello "+self.name+"!")
	def goodbye(self):
		print("Goodbye "+self.name+"!")
		
m = Man("Wayne")
m.hello()
m.goodbye()
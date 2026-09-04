import matplotlib.pyplot as plt
marks=[90,80,70,200,450,300,400,500,600,700]
plt.hist(marks,bins=5,edgecolor='black',color='blue')
plt.title("Student Marks Histogram")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()
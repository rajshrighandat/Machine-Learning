

import matplotlib.pyplot as plt 
subject=["ML","Dl","DS"]
Mark=[90,80,70]
plt.pie(Mark, labels=subject, autopct='%1.17f%%', startangle=30)
plt.title("Student Marks")
plt.xlabel("subject")
plt.ylabel("Marks")
plt.show()


if __name__=='__main__':
    from tkinter import *
    import minor
    def number():
        global b
        b = int(f.get())
    root2 = Tk()
    root2.title('Prediction App')
    myLabel2 = Label(root2, text = "How many days would you like to see in the future?\n(Ref Date: 2022-4-16)" ,font = ("Helvetica", 10, "bold italic"))
    myLabel2.pack()
    f = Entry(root2)
    f.pack()
    button_pre = Button(root2, text = 'Confirm the number', padx = 20 , pady = 20, command=number)
    button_pre.pack()
    button_22= Button(root2, text = 'See the results', padx = 20 , pady = 20, command  = lambda : minor.actualui(b))
    button_22.pack()
    root2.mainloop()
    global b
    pred_days = b



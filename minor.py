from tkinter import *

def actualui(days):
    def bitcoin():
        import prediction_function_btc
        root = Tk()
        root.title('Bitcoin Prices')
        myLabel = Label(root, text = "What Algorithm's performance would you like to see?" ,font =("Helvetica", 10, "bold italic"))
        myLabel.pack()
        button_1 = Button(root, text = 'LSTM' ,padx = 20 , pady = 20,  command = lambda : prediction_function_btc.model_selection('btc_lstm',days))
        button_1.place(x = 0, y = 100)
        button_2 = Button(root, text = 'GRU', padx = 20 , pady = 20,  command = lambda : prediction_function_btc.model_selection('btc_gru',days))
        button_2.place(x = 85, y = 100)
        root.geometry("640x480")
        root.mainloop()
        return
    def litecoin():
        root = Tk()
        root.title('Litecoin Prices')
        myLabel = Label(root, text = "What Algorithm's performance would you like to see?" ,font =("Helvetica", 10, "bold italic"))
        myLabel.pack()
        import prediction_function_ltc
        button_1 = Button(root, text = 'LSTM' ,padx = 20 , pady = 20,  command = lambda : prediction_function_ltc.model_selection('ltc_lstm',days))
        button_1.place(x = 0, y = 100)
        button_2 = Button(root, text = 'GRU', padx = 20 , pady = 20,  command = lambda : prediction_function_ltc.model_selection('ltc_gru',days))
        button_2.place(x = 85, y = 100)
        root.geometry("640x480")
        root.mainloop()
        return
    def ETH():
        root = Tk()
        root.title('Ethereum Prices')
        myLabel = Label(root, text = "What Algorithm's performance would you like to see?" ,font =("Helvetica", 10, "bold italic"))
        myLabel.pack()
        import prediction_function_eth
        button_1 = Button(root, text = 'LSTM' ,padx = 20 , pady = 20,  command =lambda : prediction_function_eth.model_selection('ETH_lstm',days))
        button_1.place(x = 0, y = 100)
        button_2 = Button(root, text = 'GRU', padx = 20 , pady = 20,  command = lambda : prediction_function_eth.model_selection('ETH_gru',days))
        button_2.place(x = 85, y = 100)
        root.geometry("640x480")
        root.mainloop()
        return


    root = Tk()
    root.title('Prediction App')
    myLabel = Label(root, text = "SELECT CRYPTOCURRENCY" ,font =("Helvetica", 20, "bold italic"))
    myLabel.pack()
    button_1 = Button(root, text = 'Bitcoin' ,padx = 20 , pady = 20, command=bitcoin)
    button_1.place(x = 0, y = 100)
    button_2 = Button(root, text = 'ETH', padx = 20 , pady = 20, command=ETH)
    button_2.place(x = 85, y = 100)
    button_3 = Button(root, text = 'Litecoin', padx = 20 , pady = 20, command=litecoin)
    button_3.place(x = 155, y = 100)
    root.geometry("640x480")
    root.mainloop()
    return


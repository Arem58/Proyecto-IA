from tkinter import * 
from tkinter import filedialog
import os
import tkinter as tk 
from PIL import Image, ImageTk #python3-pil.imagetk python3-imaging-tk
from img import *

path = ''

def showimage():
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File", filetypes=(("JPG File", "*.jpg"), ("PNG File", "*.png"), ("All File", "*.*")))
    img = Image.open(fln)
    img.thumbnail((350,350))
    img = ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image = img
    app(fln)
    

window = Tk()

frm = Frame(window)
frm.pack(side=BOTTOM, padx=15, pady=15)

lbl = Label(window)
lbl.pack()

btn = Button(frm, text="Browse Image", command=showimage)
btn.pack(side=tk.LEFT)


btn2 = Button(frm, text="Exit", command=lambda: exit())
btn2.pack(side=tk.BOTTOM, padx=10)

window.title("Image Browser")
window.geometry("300x350")
window.mainloop()
#!/usr/bin/env python3 
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.constants import *
from tkinter import filedialog
from PIL import ImageTk, Image
import webbrowser, os, re, pathlib

import seg

class Application(ttk.Frame):

  @classmethod
  def main(cls):
    root = tk.Tk()
    app = cls(root)
    root.title("Categorizer v0.1")
    app.grid(row=0, column=0, sticky=NSEW)
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    root.resizable(True, True)
    root.mainloop()

  def __init__(self, root):
    width = root.winfo_screenwidth() // 2
    height = root.winfo_screenheight() // 2
    super().__init__(root, padding=(3,3,12,12))
    
    # Change to tk.PhotoImage(file=filename) when packaging up
    # on OSX, tk 8.5 doesn't support png
    self.icons = {
      #'info': ImageTk.PhotoImage(Image.open("question16.png"))
      'info': tk.PhotoImage(file='question16.png')
    }

    self.create_variables()
    self.create_widgets()
    for i in range(3):
      self.grid_rowconfigure(i, weight=1)
      self.grid_columnconfigure(i, weight=1)
    self.logfile = open('debug.log', 'w')

  def create_variables(self):
    self.enc_type = tk.StringVar(self, 'Encoding')
    self.enc_vals = ['TFIDF', 'BoW']

  def make_scrolled(self, WidgetClass, frame, clsargs, gridargs):
    # let's make this in a frame so the scrollbars don't mess up the grid
    container = ttk.Frame(frame, padding=(0,0,0,0),
                          borderwidth=0)
    container.grid(**gridargs)
        
    widget = WidgetClass(container, **clsargs)
    widget.grid(row=0, column=0, sticky='nsew')
        
    vscroll = ttk.Scrollbar(container, orient='vertical',
                            command=widget.yview)
    vscroll.grid(row=0, column=1, rowspan=2, sticky='nse')
    widget.config(yscrollcommand=vscroll.set)

    hscroll = ttk.Scrollbar(container, orient='horizontal',
                            command=widget.xview)
    hscroll.grid(row=1, column=0, columnspan=2, sticky='ews')
    widget.config(xscrollcommand=hscroll.set)

    return widget

  def make_frame(self, row, col, args={}):
    if 'padding' not in args: args['padding'] = (3,3,12,12)
    frame = ttk.Frame(self, **args)
    frame.grid(row=row, column=col, sticky='nsew')
    return frame

  def make_file_frame(self, row, col):
    frame = self.make_frame(row, col)

    but = ttk.Button(frame, text='Add files',
                     command=self.add_files)
    but.grid(column=0, row=0, sticky='ew')
    but = ttk.Button(frame, text='Remove files',
                     command=self.remove_files)
    but.grid(column=1, row=0, sticky='ew')

    gridargs = {'column': 0, 'row': 1, 'columnspan':2,
                'sticky':'nsew'}
    clsargs = {'selectmode':'extended', 'width': 53}
    self.w_files = self.make_scrolled(tk.Listbox, frame,
                                      clsargs, gridargs)

  def make_output(self, row, col):
    frame = self.make_frame(row, col)

    but = ttk.Button(frame, text='Open file',
                     command=self.open_file)
    but.grid(column=0, row=0, sticky='ew')
    but = ttk.Button(frame, text='Open folder',
                     command=self.open_folder)
    but.grid(column=1, row=0, sticky='ew')

    gridargs = {'column': 0, 'row': 1, 'columnspan':2,
                'sticky':'nsew'}
    clsargs = {'width': 62, 'height':18}
    self.w_output = self.make_scrolled(tk.Listbox, frame,
                                       clsargs, gridargs)

  def single_option(self, frame, row, labeltxt, WidgetClass, helpstring=None, clsargs={}):
    l = ttk.Label(frame, text=f"{labeltxt}: ")
    l.grid(row=row, column=0, sticky='w')
    w = WidgetClass(frame, **clsargs)
    w.grid(row=row, column=1, sticky='e')

    if helpstring is not None:
      b = ttk.Button(frame, image=self.icons['info'],
                     command=lambda: self.info_popup(labeltxt, helpstring))
      b.grid(row=row, column=2)
    return l,w

  def make_options(self, row, col):
    regex_description = """
^\t=\tStart of line,
$\t=\tEnd of line,
\\w\t=\tAny letter, number or underscore,
+\t=\tRepeat of previous thing one or more times
"""
    
    frame = self.make_frame(row, col, {'relief':'sunken', 'borderwidth':5})
    title = ttk.Label(frame, text="Options")
    title.grid(row=0, column=0, columnspan=2)
    
    _, self.p_doc_split = self.single_option(frame, 1, 'Article Split String', ttk.Entry,
                                             regex_description)
    self.p_doc_split.insert(END, '^Document \w+$')
    _, self.p_md_split = self.single_option(frame, 2, 'Metadata Split String', ttk.Entry,
                                            regex_description)
    self.p_md_split.insert(END, '^English$')
    _, self.p_nclusters = self.single_option(frame, 3, 'Num Clusters', ttk.Entry)
    self.p_nclusters.insert(END, '10')
        
    comboargs = {'textvariable':self.enc_type, 'state':'readyonly', 'values':self.enc_vals}
    _, self.p_encoding = self.single_option(frame, 4, 'Encoding', ttk.Combobox, 'TODO', comboargs)
    self.p_encoding.set(self.enc_vals[0])
    
    _, self.p_nclosest = self.single_option(frame, 5, 'Closest K Articles, K', ttk.Entry)
    self.p_nclosest.insert(END, '3')

    but = ttk.Button(frame, text="Go!", command=self.go)
    but.grid(row=6, column=0, columnspan=2)

    return frame

  def make_status(self, row, col):
    frame = self.make_frame(row, col)
    logargs = {'state':'disabled', 'background':'lightgray',
               'width':40, 'height':20}
    gridargs = {'row':0, 'column':0, 'sticky':'nsew'}
    self.w_log = self.make_scrolled(tk.Text, frame, logargs, gridargs)
    return frame

  def create_widgets(self):
    self.f_status = self.make_status(1, 0)
    self.f_files = self.make_file_frame(0,0)
    self.f_options = self.make_options(0, 1)
    self.f_output = self.make_output(1,1)

  def log(self, string, end='\n'):
    self.w_log.config(state='normal')
    self.w_log.insert('end', string + end)
    self.w_log.config(state='disabled')
    print(string + end, end='', file=self.logfile)
    print(string + end, end='')
      
  def go(self):
    params = {'log': self.log}
    params['article_splitter'] = re.compile(self.p_doc_split.get(), re.MULTILINE)
    params['metadata_splitter'] = re.compile(self.p_md_split.get(), re.MULTILINE)
    ncluster = int(self.p_nclusters.get())
    params['num_clusters'] = ncluster

    encoding = self.p_encoding.get()
    if encoding == 'TFIDF':
      params['vectorizer'] = seg.TfidfVectorizer
    else:
      assert encoding == 'BoW'
      params['vectorizer'] = seg.CountAvgVectorizer

    paths = list(self.w_files.get(0, tk.END))
    if len(paths) == 0:
      self.log("No input files!")
      return

    cat = seg.TextCategorizer(**params)
    cat.load_files(paths)
    cat.fit()

    k = int(self.p_nclosest.get())
    near = cat.nearest(k)
    pathlib.Path("./output/").mkdir(parents=True, exist_ok=True)
    for i in range(ncluster):
      filename = f"output/cluster{i}.txt"
      with open(filename, 'w') as f:
        f.write(f"----- Cluster {i} -----\n")
        for j in range(k):
          f.write(near[i][j])
          f.write("\n -------------------- \n")
      self.w_output.insert(END, filename)


    

  def info_popup(self, labeltxt, helptxt):
    t = tk.Toplevel(self)
    t.title(f'{labeltxt} Info')
    t.resizable(False, False)
    t.lift(self)
    lab = ttk.Label(t, text=helptxt)
    lab.grid()
    

  def about(self):
    t = tk.Toplevel(self)
    t.title('About')
    t.resizable(False, False)
    t.lift(self)

  def open_file(self):
    fileno = self.w_output.curselection()
    fpath = self.w_output.get(fileno)
    webbrowser.open(os.path.realpath(fpath))

  def open_folder(self):
    webbrowser.open(os.path.realpath('./output/'))
      
  def add_files(self):
    files = filedialog.askopenfilenames(initialdir='.',
                                        title='Select files',
                                        filetypes = (("text files", "*.txt"),
                                                     ("all files", "*.*")))
    for f in files:
      self.w_files.insert(END, f)

  def remove_files(self):
    filenos = self.w_files.curselection()
    # A little tricky, since each deletion changes the indices...
    length = self.w_files.size()
    newfiles = [self.w_files.get(i) for i in range(length) if i not in filenos]
    self.w_files.delete(0, END)
    for fname in newfiles:
      self.w_files.insert(END, fname)

if __name__ == '__main__':
  Application.main()

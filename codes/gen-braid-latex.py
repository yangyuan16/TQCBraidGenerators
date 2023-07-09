
def braid_to_latex(n, sequence):
    code = ""
    nb_layers = len(sequence)
    for nn in range(n):
        code += f"\draw ({nn*2},0) circle (0pt) node[anchor=north]{{${nn}$}};\n"
        
    for l, braid in enumerate(sequence):
       i, inv = braid[0], braid[1]
       for nn in range(n):
           if nn == i and inv == 0:
               code += "% braid\n"
               code += f"\draw[thick] ({i*2},{l*5}) .. controls ({i*2},{l*5+2}) and ({i*2+2},{l*5+3}) .. ({i*2+2},{l*5+5});\n"
               code += f"\draw[thick] ({i*2+2},{l*5}) .. controls ({i*2+1.7},{l*5+1.5}) .. ({i*2+1.2},{l*5+2.3});\n"
               code += f"\draw[thick] ({i*2+0.8},{l*5+2.7}) .. controls ({i*2+0.3},{l*5+3.5}) .. ({i*2},{l*5+5});\n"
           elif nn == i and inv == 1:
               code += "% braid\n"
               code += f"\draw[thick] ({i*2+2},{l*5}) .. controls ({i*2+2},{l*5+2}) and ({i*2},{l*5+3}) .. ({i*2},{l*5+5});\n"
               code += f"\draw[thick] ({i*2},{l*5}) .. controls ({i*2+0.3},{l*5+1.5}) .. ({i*2+0.8},{l*5+2.3});\n"
               code += f"\draw[thick] ({i*2+1.2},{l*5+2.7}) .. controls ({i*2+1.7},{l*5+3.5}) .. ({i*2+2},{l*5+5});\n"
           elif nn == i+1:
               continue
           else:
               code += f"\draw[thick] ({nn*2},{l*5}) -- ({nn*2},{l*5+5});\n"
        
    print(code)


"""
\begin{tikzpicture}[x=0.4cm, y=0.4cm]
        \draw[thick] (0,0) node[anchor=north]{$1$} -- (0,5);
        \draw[thick] (2,0) node[anchor=north]{$2$} -- (2,5);
        % dots
        \filldraw (3, 2.5) circle (0.5pt);
        \filldraw (3.5, 2.5) circle (0.5pt);
        \filldraw (4, 2.5) circle (0.5pt);
        % braid
        \draw[thick] (5,0) node[anchor=north]{$i$} .. controls (5,2) and (7,3) .. (7,5);
        \draw[thick] (7,0) node[anchor=north]{$i+1$} .. controls (6.7,1.5) .. (6.2,2.3);
        \draw[thick] (5.8,2.7) .. controls (5.3,3.5) .. (5,5);
        % dots
        \filldraw (8, 2.5) circle (0.5pt);
        \filldraw (8.5, 2.5) circle (0.5pt);
        \filldraw (9, 2.5) circle (0.5pt);
        % last lines
        \draw[thick] (10,0) node[anchor=north]{$n-1$} -- (10,5);
        \draw[thick] (12,0) node[anchor=north]{$n$} -- (12,5);
\end{tikzpicture}
"""
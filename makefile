cnot: codes/cnot.py
	PYTHONPATH=. python codes/cnot.py

clean:
	rm -f *.ps *.log *.aux *.out *.dvi *.bbl *.blg
	rm -f images/*
	rm -f bin/*

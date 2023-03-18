foo: ;
	cat gnn/gnn.md | sed /^\\$$\\$$/d | pandoc -f markdown -s -o gnn/gnn.pdf


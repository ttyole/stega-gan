all:
	mkdir build
	g++ `python -m pybind11 --includes` -I/usr/include/python2.7 -I includes -O3 -std=c++11 -c -fPIC -o build/STC.o ml_stc_src/STC.cpp
	g++ -I includes -std=c++98 -O3 -c -fPIC -o build/stc_embed_c.o 		ml_stc_src/stc_embed_c.cpp
	g++ -I includes -std=c++98 -O3 -c -fPIC -o build/stc_extract_c.o 	ml_stc_src/stc_extract_c.cpp
	g++ -I includes -std=c++98 -O3 -c -fPIC -o build/common.o 			ml_stc_src/common.cpp
	g++ -I includes -std=c++98 -O3 -c -fPIC -o build/stc_ml_c.o 		ml_stc_src/stc_ml_c.cpp
	g++ -O3 -shared -Wall build/STC.o build/stc_embed_c.o build/stc_extract_c.o build/common.o build/stc_ml_c.o -o STC.so
	make clean
clean:
	rm -f build/*
	rmdir build

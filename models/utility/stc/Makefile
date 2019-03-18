all:
	g++ `python3 -m pybind11 --includes` -I includes -O3 -std=c++11 -c -fPIC -o release/STC.o ml_stc_src/STC.cpp
	g++ -I includes -std=c++98 -O3 -c -fPIC -o release/stc_embed_c.o 	ml_stc_src/stc_embed_c.cpp
	g++ -I includes -std=c++98 -O3 -c -fPIC -o release/stc_extract_c.o 	ml_stc_src/stc_extract_c.cpp
	g++ -I includes -std=c++98 -O3 -c -fPIC -o release/common.o 		ml_stc_src/common.cpp
	g++ -I includes -std=c++98 -O3 -c -fPIC -o release/stc_ml_c.o 		ml_stc_src/stc_ml_c.cpp
	g++ -O3 -shared -Wall release/STC.o release/stc_embed_c.o release/stc_extract_c.o release/common.o release/stc_ml_c.o -o release/STC.so
clean:
	rm -f release/*

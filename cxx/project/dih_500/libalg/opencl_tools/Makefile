SUBDIRS=$(shell ls -l | grep ^d | awk '{print $$9}')
CUR_SOURCE=${wildcard *.cpp}
CUR_OBJS=${patsubst %.cpp, %.o, $(CUR_SOURCE)}
all:$(SUBDIRS) $(CUR_OBJS)
$(SUBDIRS):ECHO
	make -C $@ -j8
$(CUR_OBJS):%.o:%.cpp
	$(CXX) -c $^ -o $(OBJ_PATH)/$@ $(CPPFLAGS)
ECHO:
	@echo $(SUBDIRS)
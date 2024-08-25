What I used to Build (windows)

```sh
del /F /Q *  
cmake -G "Visual Studio 17 2022" ../src
cmake --build . --config Release -j 4
```

Should be able to work w/linux via base instructions. 

To work with python, need to build in release (unless you have debug python setup which is a pain in the ass)
# INSTRUCTIONS TO INSTALL PYTORCH3D ON WINDOWS

- Download nvidia_cub and set env variable CUB_HOME=<cub_path>
- Clone pytorch3d repository from github
- Install Visual Studio and add path to cl.exe to system path
- Use pytorch==1.6.0

# PATCHES TO PYTORCH FILES WHEN BUILDING ON WINDOWS
Change python/Lib/site-packages/torch/include/torch/csrc/jit/api/module.h

L483, 496, 510, 523
```
-static constexpr
+static const
```
Change python/Lib/site-packages/torch/include/torch/csrc/jit/runtime/argument_spec.h

L161
```
-static CONSTEXPR_EXCEPT_WIN_CUDA size_t ARG_SPEC_DEPTH_LIMIT = 128;
+static const size_t ARG_SPEC_DEPTH_LIMIT = 128;
```

Change python/Lib/site-packages/torch/include/pybind11/cast.h

L1449
```
-explicit operator type&() { return *(this->value); }
+explicit operator type& () { return *((type*)(this->value)); }
```
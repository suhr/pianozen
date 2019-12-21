with import <nixpkgs> {};

stdenv.mkDerivation rec {
  name = "xephys";
  buildInputs = [
    jack2
  ];

  LD_LIBRARY_PATH = builtins.foldl'
    (a: b: "${a}:${b}/lib") "/run/opengl-driver/lib" buildInputs;
}

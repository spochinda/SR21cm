# SR21cm
Generative models to produce super-resolved 21-cm brightness temperature simulations

## Setting up on Tursa

Install Python from source and sqlite for Jupyter to work (Also see https://stackoverflow.com/questions/56737495/how-to-install-sqlite3-for-python3-7-in-seperate-directory-on-linux-without-sudo)

$ mkdir -p ~/.localapps/src
$ cd ~/.localapps/src
$ # Download and build sqlite 3 (you might want to get a newer version)
$ wget http://www.sqlite.org/sqlite-autoconf-3070900.tar.gz
$ tar xvvf sqlite-autoconf-3070900.tar.gz
$ cd sqlite-autoconf-3070900
$ ./configure --prefix=~/.localapps
$ make
$ make install

$ # Now download and build python 2, same works for python 3
$ cd ~/.localapps/src
$ wget http://www.python.org/ftp/python/3.8.9/Python-3.8.9.tgz
$ tar xvvf Python-3.8.9.tgz
$ cd Python-3.8.9
$ ./configure --prefix=~/.localapps
$ make
$ make install

$ makedir ~/venvs
$ cd 
$ .localapps/bin/python3 -m venv venvs/superres
$ source venvs/superres/bin/activate
$ pip install -r requirements.txt






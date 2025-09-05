Projekt składa się z folderów:
- Rezultaty: wykresy i arkusze z wynikami kalibracji sieci.
- Kod: modele sieci neuronowe, solwery numeryczne - w tym PINN.
- Inne: dokumenty techniczne.

Dokładny opis zawartości znajduje się w pracy magisterskiej, w części Załączniki.

Aby uruchomić symulator BQC, którego wynikiem działania są wykresy i wyniki symulacji w plikach xls, należy zainstalować symulator SquidASM oraz NetSquid na środowisku Unixowym lub WSL, zgodnie z instrukcjami:
- NetSquid: https://netsquid.org/
- SquidASM: https://squidasm.readthedocs.io/en/latest/installation.html

Po zainstalowaniu obu pakietów, należy uruchomić poniższe komendy:
- cd \Dodatek\Kod\Serwer kwantowy\BQC - NetSquid
- python bqc.py

Wynikiem działania skryptu powinien być zestaw pomiarów. Ze względu na wartość ustawioną domyślnie, symulacja może zająć kilka godzin. Aby zmniejszyć czas potrzebny na realizację pomiarów (ale też ich dokładność), należy zmienić wartość num_times = 1000 w linii 241 na np. 10.

Wygenerowane wcześniej wyniki eksperymentów dostępne są w folderze głównym oraz "BQC/graphs".


W celu uruchomienia sieci qGAN, należy przejść do folderu "Dodatek\Kod\Klient kwantowy\qGAN - Classifier". Uruchomienie znajdujących się tam modeli wymaga zainstalowania pakietów, które zostały wskazane w pierwszej komórce programu.

Kod AMD i pozostałych programów znajdujących się w folderze Dodatek\Kod\ można uruchomić analogicznie do powyższej instrukcji.
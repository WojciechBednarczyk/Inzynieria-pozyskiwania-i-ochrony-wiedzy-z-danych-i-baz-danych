    Wczytanie Obrazu: Na początku obraz jest wczytywany z podanej ścieżki. Obraz ten jest wczytywany jako obraz w odcieniach szarości.

    Zastosowanie Maski Tła: Tworzona jest maska tła, która zakrywa obszary tła na obrazie. Maskowanie tła jest wykonywane poprzez zastosowanie maski, która jest tworzona na podstawie wartości progowej. Obszary o jasności mniejszej niż wartość progu są traktowane jako tło i zostają zakryte na obrazie.

    Rozmycie Gaussowskie: Na zakrytym obrazie jest stosowane rozmycie Gaussowskie z określonym rozmiarem jądra rozmycia. Rozmycie Gaussowskie pomaga w wygładzeniu obrazu i redukcji szumów.

    Progowanie: Obraz jest progowany, co oznacza przypisanie wartości pikselom na podstawie pewnego progu. Wartość progowa może być wybierana na różne sposoby, w zależności od ustawień. Może to być progowanie otsu, progowanie adaptacyjne lub progowanie na podstawie histogramu. Wynikiem jest obraz binarny, gdzie piksele są albo czarne (jeśli przekroczyły próg) albo białe (jeśli nie przekroczyły).

    Usuwanie Szumów: Na obrazie binarnym jest stosowana operacja morfologiczna otwarcia, która pomaga w usunięciu małych zakłóceń lub szumów na obrazie binarnym.

    Wyszukiwanie Konturów: Następnie są wyszukiwane kontury na przetworzonym obrazie binarnym. Kontury reprezentują różne obiekty lub regiony na obrazie.

    Rysowanie Konturów Płuc: Na podstawie obszaru konturów i zdefiniowanego progu obszaru, kontury reprezentujące płuca są wyodrębniane. To jest etap identyfikacji obszarów reprezentujących płuca.

    Kolorowanie Obrazów: Obrazy są konwertowane na kolorowe obrazy, gdzie oryginalny obraz jest pozostawiony w odcieniach szarości, a obszar płuc jest kolorowany na czerwono dla lepszej widoczności.

    Nakładanie Obrazów: Kolorowe obrazy płuc są nakładane na oryginalny obraz, tworząc finalny wynik. Płuca są widoczne na oryginalnym obrazie jako czerwone obszary.

    Przygotowanie Wyniku: Finalny obraz jest przygotowywany poprzez połączenie oryginalnego obrazu i obrazu z zaznaczonymi płucami.

    Zapis Wyniku: Wynik jest zapisywany w określonej lokalizacji.
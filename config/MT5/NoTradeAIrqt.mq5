// Importation de la librairie WinINet pour les requêtes HTTP
#import "wininet.dll"
int InternetOpenW(string, int, string, string, int);
int InternetConnectW(int, string, int, string, string, int, int, int);
int HttpOpenRequestW(int, string, string, string, string, string, int, int);
int HttpSendRequestW(int, string, int, uchar &lpData[], int);
int InternetReadFile(int, uchar&, int, int&);
int InternetCloseHandle(int);
#import

// Déclaration globale pour les requêtes
int handleInternet, handleConnection, handleRequest;

// Structur e pour stocker la réponse JSON
struct TradeData {
    double SL;
    double TP;
    string order_type;
    string request_id;
    string symboles;  // Correction ici, symboles au lieu de symbole
    double predicted_close;
    string statut;
};
struct TradeInfo {
    string symbol;
    bool isOpen;
};

TradeInfo tradesInProgress[];

static datetime lastRequestTime = 0; // Variable statique pour garder une trace du dernier envoi
const int requestInterval = 1; // Intervalle en secondes entre les requêtes

void OnTick() {
    string symbol = Symbol();
    datetime currentTime = TimeCurrent();

    if (currentTime - lastRequestTime >= requestInterval) {
        lastRequestTime = currentTime;

        string jsonData = GetPriceData();

        //Print("Données envoyées : ", jsonData);
        CloseAllPositions();
        GetVolumeFromHTML();
        string jsonResponse = SendHttpRequest("POST", "/receive_data", jsonData);
        // Print("Réponse JSON reçue : ", jsonResponse);

        if (StringLen(jsonResponse) > 0) {
            TradeData tradeData;
            if (ParseResponse(jsonResponse, tradeData)) {
                //  Print("Réponse JSON analysée : ", tradeData.statut, ", SL: ", tradeData.SL, ", TP: ", tradeData.TP, ", order_type: ", tradeData.order_type);
                // Print("Symbole récupéré : ", symbol);

                if (tradeData.statut == "1") {
                    if (!IsPositionOpen(symbol)) {
                        // Vérifier si les trades sont bloqués
                        if (IsTradingBlocked()) {
                            Print("Les trades sont bloqués.");
                            return;
                        } else {
                            
                        }
                    } else {
                        Print("Trade déjà ouvert ou symbole bloqué pour : ", symbol);
                    }
                } else {
                    Print("Échec de l'ouverture du trade, statut : ", symbol);
                }
            } else {
                Print("Échec de l'analyse de la réponse JSON.");
            }
        } else {
            Print("La réponse JSON est vide ou invalide.");
        }
    }
}

// Fonction pour récupérer les données de prix
string GetPriceData() {
    datetime current_time = TimeCurrent();
    MqlDateTime timeStruct;
    TimeToStruct(current_time, timeStruct); // Remplir la structure avec les valeurs de temps

    string dateheure = StringFormat("%04d-%02d-%02d %02d:%02d:%02d",
                                    timeStruct.year,
                                    timeStruct.mon,
                                    timeStruct.day,
                                    timeStruct.hour,
                                    timeStruct.min,
                                    timeStruct.sec);

    // Correction de l'accès aux prix de clôture
    double prix = NormalizeDouble(iClose(Symbol(), 0, 0), _Digits); // Utiliser iClose pour accéder aux prix de clôture
    string symbole = Symbol(); // Récupérer le symbole actuel

    // Construction correcte du JSON avec le symbole
    string jsonData = "{\"datetime\": \"" + dateheure + "\", \"prix\": \"" + DoubleToString(prix, _Digits) + "\", \"symboles\": \"" + symbole + "\"}";

    return jsonData;
}

string SendHttpRequest(string requestType, string urlPath, string jsonData = "") {
    string result = "";

    // Ouvrir une connexion Internet
    handleInternet = InternetOpenW("MT5", 1, NULL, NULL, 0);
    if (handleInternet == 0) {
        Print("Erreur lors de l'ouverture de la connexion Internet : ", IntegerToString(GetLastError()));
        return result;
    }

    handleConnection = InternetConnectW(handleInternet, "127.0.0.1", 5000, NULL, NULL, 3, 0, 0);
    if (handleConnection == 0) {
        Print("Erreur lors de la connexion au serveur : ", IntegerToString(GetLastError()));
        InternetCloseHandle(handleInternet);
        return result;
    }

    handleRequest = HttpOpenRequestW(handleConnection, requestType, urlPath, NULL, NULL, NULL, 0x04000000, 0);
    if (handleRequest == 0) {
        Print("Erreur lors de l'ouverture de la requête : ", IntegerToString(GetLastError()));
        InternetCloseHandle(handleConnection);
        InternetCloseHandle(handleInternet);
        return result;
    }

    uchar utf8Data[4096];
    int dataSize = StringToCharArray(jsonData, utf8Data);

    if (dataSize == 0) {
        Print("Erreur lors de la conversion de la chaîne JSON en tableau de caractères.");
        InternetCloseHandle(handleRequest);
        InternetCloseHandle(handleConnection);
        InternetCloseHandle(handleInternet);
        return result;
    }

    string headers = "Content-Type: application/json; charset=UTF-8";
    int requestResult = HttpSendRequestW(handleRequest, headers, StringLen(headers), utf8Data, dataSize);
    if (requestResult == 0) {
        int error = GetLastError();
        Print("Erreur lors de l'envoi de la requête : ", IntegerToString(error));
        InternetCloseHandle(handleRequest);
        InternetCloseHandle(handleConnection);
        InternetCloseHandle(handleInternet);
        return result;
    }

    uchar buffer[8192];
    int bytesRead = 0;

    while (InternetReadFile(handleRequest, buffer[0], sizeof(buffer), bytesRead) && bytesRead > 0) {
        result += CharArrayToString(buffer, 0, bytesRead, CP_UTF8);
    }

    //Print("Réponse brute lue : ", result);

    if (StringLen(result) == 0) {
        Print("Erreur : la réponse est vide.");
    }

    InternetCloseHandle(handleRequest);
    InternetCloseHandle(handleConnection);
    InternetCloseHandle(handleInternet);

    return result;
}

bool ParseResponse(string jsonResponse, TradeData &tradeData) {
    if (StringLen(jsonResponse) == 0) {
        Print("La réponse JSON est vide.");
        return false;
    }

    // Extraction avec vérification
    tradeData.SL = StringToDouble(JsonGetValue(jsonResponse, "SL"));
    tradeData.TP = StringToDouble(JsonGetValue(jsonResponse, "TP"));
    tradeData.order_type = JsonGetValue(jsonResponse, "order_type");
    tradeData.symboles = JsonGetValue(jsonResponse, "symboles"); // Extraction ajoutée ici
    tradeData.predicted_close = StringToDouble(JsonGetValue(jsonResponse, "predicted_close"));
    tradeData.request_id = JsonGetValue(jsonResponse, "request_id");
    tradeData.statut = JsonGetValue(jsonResponse, "statut");

    // Vérification
    if (tradeData.statut == "" || tradeData.order_type == "") {
        Print("Erreur dans les données reçues : statut ou type de commande manquant.");
        return false;
    }

    if (tradeData.SL <= 0.0 || tradeData.TP <= 0.0) {
        Print("Erreur : SL ou TP invalide.");
        return false;
    }

    return true;
}

// Fonction pour supprimer les espaces en début et fin de chaîne en MQL5
string Trim(string str) {
    int len = StringLen(str);
    int start = 0, end = len - 1;

    // Supprimer les espaces au début
    while (start < len && StringGetCharacter(str, start) == ' ')
        start++;

    // Supprimer les espaces à la fin
    while (end > start && StringGetCharacter(str, end) == ' ')
        end--;

    return StringSubstr(str, start, end - start + 1);
}

// Fonction pour extraire une valeur d'une réponse JSON
string JsonGetValue(string json, string key) {
    int start = StringFind(json, "\"" + key + "\":");
    if (start == -1) {
        Print("Clé non trouvée dans la réponse JSON : ", key);
        return ""; // Clé non trouvée
    }
    start += StringLen(key) + 3; // Passer le `":`

    // Vérifier le premier caractère après le ": pour identifier les chaînes, nombres, etc.
    string valueType = StringSubstr(json, start, 1);
    int end;
    if (valueType == "\"") {
        // Valeur chaîne
        start++; // Passer le guillemet
        end = StringFind(json, "\"", start);
    } else {
        // Valeur nombre ou booléen
        end = StringFind(json, ",", start);
        if (end == -1) end = StringFind(json, "}", start); // Fin du JSON
    }

    return Trim(StringSubstr(json, start, end - start));
}

bool IsPositionOpen(string symbol) {
    // Vérifie directement si une position est ouverte pour le symbole donné
    if (PositionSelect(symbol)) {
        return true; // Une position est ouverte pour ce symbole
    }
    return false;
}

void OpenTrade(TradeData &tradeData) {
    string symbol = Symbol();
    double sl = NormalizeDouble(tradeData.SL, _Digits); // Stop Loss
    double tp = NormalizeDouble(tradeData.TP, _Digits); // Take Profit
    double volume = GetVolumeFromHTML(); // Récupérer le volume depuis la page HTML

    // Vérifiez que le volume est valide
    if (volume < SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN)) {
        Print("Erreur : Volume en dessous du minimum autorisé.");
        return;
    }

    MqlTradeRequest request;
    MqlTradeResult result;
    ZeroMemory(request);
    ZeroMemory(result);

    request.action = TRADE_ACTION_DEAL; // Action de trading
    request.symbol = symbol;
    request.volume = volume; // Volume de la commande
    request.magic = 123456; // Identifiant magique pour la commande

    // Récupérer le prix actuel pour l'ordre
    if (tradeData.order_type == "1") {
        request.type = ORDER_TYPE_BUY; // Type d'ordre
        request.price = SymbolInfoDouble(symbol, SYMBOL_BID); // Prix d'achat
    } else {
        request.type = ORDER_TYPE_SELL; // Type d'ordre
        request.price = SymbolInfoDouble(symbol, SYMBOL_ASK); // Prix de vente
    }

    // Vérifiez que le prix n'est pas 0
    if (request.price == 0) {
        Print("Erreur : Le prix récupéré est nul pour le symbole : ");
        return; // Arrêtez la fonction si le prix est invalide
    }

    // Vérification de SL et TP
    if (sl <= 0 || tp <= 0) {
        Print("Erreur : SL ou TP invalide.");
        return;
    }

    request.tp = tp; // Définir le Take Profit
    request.sl = sl; // Définir le Stop Loss

    // Journaliser les détails de l'ordre
    //Print("Ouverture de trade : symbole: ", request.symbol, ", type: ", request.type, ", prix: ", request.price, ", SL: ", sl, ", TP: ", tp);

    // Envoi de la requête d'ouverture de trade
    if (!OrderSend(request, result)) {
        Print("Erreur lors de l'envoi de la commande : ", IntegerToString(GetLastError()));
    } else {
        //Print("Commande envoyée avec succès, ticket : ", result.order);

        // Mettre à jour la structure des trades en cours
        TradeInfo newTrade;
        newTrade.symbol = symbol;
        newTrade.isOpen = true;
        ArrayResize(tradesInProgress, ArraySize(tradesInProgress) + 1);
        tradesInProgress[ArraySize(tradesInProgress) - 1] = newTrade;
    }
}

// Fonction pour récupérer le volume depuis la page HTML
double GetVolumeFromHTML() {
    string jsonResponse = SendHttpRequest("GET", "/get_volume", "");
    if (StringLen(jsonResponse) > 0) {

        double volume = StringToDouble(JsonGetValue(jsonResponse, "volume"));
        Print(volume);
        if (volume > 0) {
            return volume;
        }
    }
    return 0.01; // Valeur par défaut si aucune donnée n'est récupérée
}

// Fonction pour vérifier le statut de blocage pour un symbole


// Fonction pour vérifier si les trades sont bloqués
bool IsTradingBlocked() {
    string jsonResponse = SendHttpRequest("GET", "/set_block", "");
    if (StringLen(jsonResponse) > 0) {
        string status = JsonGetValue(jsonResponse, "blocked");
        Print("IsTradingBlocked: Réponse JSON reçue : ", jsonResponse);
        Print("IsTradingBlocked: Statut récupéré : ", status);
        Print(StringToDouble(status));
        if (StringToDouble(status) == 1.0) {
            return true;
        }
    }
    return false;
}


void CloseAllPositions() {
    string jsonResponse = SendHttpRequest("GET", "/get_status");
    if (StringLen(jsonResponse) > 0) {
        string status = StringToDouble(JsonGetValue(jsonResponse, "status"));
        if (status == 404.0) { // Comparer avec une chaîne de caractères
            Print("Supp en cours");
            for (int i = PositionsTotal() - 1; i >= 0; i--) {
                if (PositionSelect(PositionGetSymbol(i))) { // Sélection de la position par symbole
                    MqlTradeRequest request;
                    MqlTradeResult result;
                    ZeroMemory(request);
                    ZeroMemory(result);

                    request.action = TRADE_ACTION_DEAL; // Utiliser DEAL pour fermer la position
                    request.position = PositionGetInteger(POSITION_TICKET); // Identifiant du ticket de position
                    request.symbol = PositionGetString(POSITION_SYMBOL); // Symbole de la position
                    request.volume = PositionGetDouble(POSITION_VOLUME); // Volume de la position
                    request.type = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY; // Type opposé pour fermeture

                    if (!OrderSend(request, result)) {
                        Print("Erreur lors de la fermeture de la position : ", IntegerToString(GetLastError()));
                    } else {
                        Print("Position fermée avec succès, ticket : ", result.order);
                        Print("Pause de 5 secondes pour éviter les trades inutile");
                        Sleep(5000);
                    }
                }
            }
        }
    }
}

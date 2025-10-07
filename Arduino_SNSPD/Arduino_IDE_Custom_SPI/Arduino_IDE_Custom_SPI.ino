#define CLK_PIN  7  // SPI Clock pin
#define MISO_PIN 6  // SPI MISO pin
#define MOSI_PIN 5  // SPI MOSI pin
#define LOAD_PIN 4  // Load signal pin

#define BAUD_RATE 115200
#define MAX_BUFFER_SIZE 34

uint8_t uartBuffer[MAX_BUFFER_SIZE];  
int bufferIndex = 0;
int DELAY_CLK = 2000; // µs
int DELAY_LOAD = 32000;
int DELAY_BYTE = 32000; 


void setup() {
    pinMode(CLK_PIN, OUTPUT);
    pinMode(MOSI_PIN, OUTPUT);
    pinMode(MISO_PIN, INPUT);  // MISO as input
    pinMode(LOAD_PIN, OUTPUT);

    digitalWrite(CLK_PIN, LOW);
    digitalWrite(LOAD_PIN, HIGH);

    Serial.begin(BAUD_RATE);
    Serial.flush();  // Clear any garbage bytes in the buffer
}

void loop() {
    Serial.flush();  // Clear any garbage bytes in the buffer
    if (Serial.available()) {
        uint8_t receivedData[MAX_BUFFER_SIZE];
        int length = 0;
        
        // Read data from Serial until 0xFF (end marker)
        while (Serial.available()) {
            uint8_t byte = Serial.read();
            if (byte == 0xFF) break;
            receivedData[length++] = byte;
            delayMicroseconds(10000);
        }

        // Call function and get response
        delayMicroseconds(1000);
        //Serial.write(receivedData, length);
        delayMicroseconds(10000);
        
    
        uint8_t* response = transmitAndReadSPI(receivedData, length);

        // Send response back to Serial
        Serial.write(response, length);
    }
}

// Sends SPI data (LSB first) and reads the response
uint8_t* transmitAndReadSPI(uint8_t* data, int length) {
    static uint8_t response[MAX_BUFFER_SIZE];  // Static so it persists after function ends

    
    delayMicroseconds(5);

    // First train: Send data
    for (int i = 0; i < length; i++) {
        shiftOutCustom(MOSI_PIN, CLK_PIN, data[i]);
    }

    delayMicroseconds(DELAY_LOAD);
    delayMicroseconds(DELAY_LOAD);
    delayMicroseconds(DELAY_LOAD);
    digitalWrite(LOAD_PIN, HIGH);
    delayMicroseconds(DELAY_LOAD);
    delayMicroseconds(DELAY_LOAD);
    digitalWrite(LOAD_PIN, LOW);
    delayMicroseconds(DELAY_LOAD);
    delayMicroseconds(DELAY_LOAD);

    // Second train: Read data
    digitalWrite(MOSI_PIN, LOW);
    for (int i = 0; i < length; i++) {
        response[i] = shiftInCustom(MISO_PIN, CLK_PIN, MOSI_PIN);
        
    }
    //digitalWrite(MOSI_PIN, HIGH);
    return response;  // Return pointer to response array
}


// Sends byte (LSB first)
void shiftOutCustom(int mosiPin, int clkPin, uint8_t val) {
    for (int i = 0; i < 8; i++) {  
        digitalWrite(mosiPin, (val & (1 << i)) ? LOW : HIGH);
        delayMicroseconds(DELAY_CLK);
        
        digitalWrite(clkPin, HIGH);
        delayMicroseconds(DELAY_CLK);
        digitalWrite(clkPin, LOW);
    }
    delayMicroseconds(DELAY_BYTE);
}

// Reads byte (LSB first)
uint8_t shiftInCustom(int misoPin, int clkPin, int mosiPin) {
    uint8_t val = 0;
    
    for (int i = 0; i < 8; i++) {
      
        delayMicroseconds(DELAY_CLK);
      
        digitalWrite(clkPin, HIGH);
        delayMicroseconds(DELAY_CLK);

        val |= (!digitalRead(misoPin)) << i;
        //Serial.println(val);
        //if (digitalRead(misoPin)) {
            //val |= (1 << i);  // Capture bit}
        
        
        digitalWrite(clkPin, LOW);
        delayMicroseconds(DELAY_CLK);
       
    }
    delayMicroseconds(DELAY_BYTE);
    return val;
}

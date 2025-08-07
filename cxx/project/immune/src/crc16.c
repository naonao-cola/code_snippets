
#include "crc16.h"

void InvertUint8(unsigned char *DesBuf, unsigned char *SrcBuf)
{
    int i;
    unsigned char temp = 0;
    
    for(i = 0; i < 8; i++)
    {
        if(SrcBuf[0] & (1 << i))
        {
            temp |= 1<<(7-i);
        }
    }
    DesBuf[0] = temp;
}

void InvertUint16(unsigned short int *DesBuf, unsigned short int *SrcBuf)  
{  
    int i;  
    unsigned short int temp = 0;    
    
    for(i = 0; i < 16; i++)  
    {  
        if(SrcBuf[0] & (1 << i))
        {          
            temp |= 1<<(15 - i);  
        }
    }  
    DesBuf[0] = temp;  
}

unsigned short int CRC16_CCITT(unsigned char *puchMsg, unsigned int usDataLen)  
{  
    unsigned short int wCRCin = 0x0000;  
    unsigned short int wCPoly = 0x1021;  
    unsigned char wChar = 0;  
    
    while (usDataLen--)     
    {  
        wChar = *(puchMsg++);  
        InvertUint8(&wChar, &wChar);  
        wCRCin ^= (wChar << 8); 
        
        for(int i = 0; i < 8; i++)  
        {  
            if(wCRCin & 0x8000)
            {
                wCRCin = (wCRCin << 1) ^ wCPoly; 
            }            
            else 
            {              
                wCRCin = wCRCin << 1;  
            }
        }  
    }  
    InvertUint16(&wCRCin, &wCRCin);  
    return (wCRCin) ;  
} 
 
unsigned short int CRC16_CCITT_FALSE(unsigned char *puchMsg, unsigned int usDataLen)  
{  
    unsigned short int wCRCin = 0xFFFF;  
    unsigned short int wCPoly = 0x1021;  
    unsigned char wChar = 0;  
    
    while (usDataLen--)     
    {  
        wChar = *(puchMsg++);  
        wCRCin ^= (wChar << 8); 
        
        for(int i = 0; i < 8; i++)  
        {  
            if(wCRCin & 0x8000)  
            {
                wCRCin = (wCRCin << 1) ^ wCPoly;  
            }
            else  
            {
                wCRCin = wCRCin << 1; 
            }            
        }  
    }  
    return (wCRCin) ;  
}  
 
unsigned short int CRC16_XMODEM(unsigned char *puchMsg, unsigned int usDataLen)  
{  
    unsigned short int wCRCin = 0x0000;  
    unsigned short int wCPoly = 0x1021;  
    unsigned char wChar = 0;  
    
    while (usDataLen--)     
    {  
        wChar = *(puchMsg++);  
        wCRCin ^= (wChar << 8);
        
        for(int i = 0; i < 8; i++)  
        {  
            if(wCRCin & 0x8000)  
            {
                wCRCin = (wCRCin << 1) ^ wCPoly;  
            }
            else
            {              
                wCRCin = wCRCin << 1;
            }
        }  
    }  
    return (wCRCin) ;  
}  
  
unsigned short int CRC16_X25(unsigned char *puchMsg, unsigned int usDataLen)  
{  
    unsigned short int wCRCin = 0xFFFF;  
    unsigned short int wCPoly = 0x1021;  
    unsigned char wChar = 0;  
    
    while (usDataLen--)     
    {  
        wChar = *(puchMsg++);  
        InvertUint8(&wChar, &wChar);  
        wCRCin ^= (wChar << 8); 
        
        for(int i = 0;i < 8;i++)  
        {  
            if(wCRCin & 0x8000)
            {              
                wCRCin = (wCRCin << 1) ^ wCPoly; 
            }            
            else  
            {
                wCRCin = wCRCin << 1; 
            }            
        }  
    }  
    InvertUint16(&wCRCin, &wCRCin);  
    return (wCRCin^0xFFFF) ;  
}  
  
unsigned short int CRC16_MODBUS(unsigned char *puchMsg, unsigned int usDataLen)  
{
	unsigned short int wCRCin= 0xFFFF; 
	unsigned short int wCPoly= 0x8005; 
	unsigned char wChar= 0;
	
    while (usDataLen--)     
    {  
			wChar = *(puchMsg++);  
			InvertUint8(&wChar, &wChar);  
			wCRCin ^= (wChar << 8); 
			
			for(int i = 0; i < 8; i++)  
			{  
					if(wCRCin & 0x8000) 
					{
							wCRCin = (wCRCin << 1) ^ wCPoly;  
					}
					else  
					{
							wCRCin = wCRCin << 1; 
					}            
			}  
    }  
    InvertUint16(&wCRCin, &wCRCin);  
    return (wCRCin) ;  
} 
 
unsigned short int CRC16_IBM(unsigned char *puchMsg, unsigned int usDataLen)  
{  
    unsigned short int wCRCin = 0x0000;  
    unsigned short int wCPoly = 0x8005;  
    unsigned char wChar = 0;  
    
    while (usDataLen--)     
    {  
        wChar = *(puchMsg++);  
        InvertUint8(&wChar, &wChar);  
        wCRCin ^= (wChar << 8);  
        
        for(int i = 0; i < 8; i++)  
        {  
            if(wCRCin & 0x8000)  
            {
                wCRCin = (wCRCin << 1) ^ wCPoly; 
            }            
            else  
            {
                wCRCin = wCRCin << 1;  
            }
        }  
    }  
    InvertUint16(&wCRCin,&wCRCin);  
    return (wCRCin) ;  
}  
 
unsigned short int CRC16_MAXIM(unsigned char *puchMsg, unsigned int usDataLen)  
{  
    unsigned short int wCRCin = 0x0000;  
    unsigned short int wCPoly = 0x8005;  
    unsigned char wChar = 0;  
    
    while (usDataLen--)     
    {  
        wChar = *(puchMsg++);  
        InvertUint8(&wChar, &wChar);  
        wCRCin ^= (wChar << 8);  
        
        for(int i = 0; i < 8; i++)  
        {  
            if(wCRCin & 0x8000) 
            {              
                wCRCin = (wCRCin << 1) ^ wCPoly;
            }
            else 
            {              
                wCRCin = wCRCin << 1;  
            }
        }  
    }  
    InvertUint16(&wCRCin, &wCRCin);  
    return (wCRCin^0xFFFF) ;  
}  
 
unsigned short int CRC16_USB(unsigned char *puchMsg, unsigned int usDataLen)  
{  
    unsigned short int wCRCin = 0xFFFF;  
    unsigned short int wCPoly = 0x8005;  
    unsigned char wChar = 0;  
    
    while (usDataLen--)     
    {  
        wChar = *(puchMsg++);  
        InvertUint8(&wChar, &wChar);  
        wCRCin ^= (wChar << 8); 
        
        for(int i = 0; i < 8; i++)  
        {  
            if(wCRCin & 0x8000) 
            {              
                wCRCin = (wCRCin << 1) ^ wCPoly;
            }            
            else  
            {
                wCRCin = wCRCin << 1; 
            }            
        }  
    }  
    InvertUint16(&wCRCin, &wCRCin);  
    return (wCRCin^0xFFFF) ;  
}



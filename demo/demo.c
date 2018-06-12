#include<stdio.h>
#include<string.h>
#include<arpa/inet.h>
#include<netdb.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<fcntl.h>
#include<sys/stat.h>
#include<stdlib.h>
#define port "80"
#define webroot "."	// Directory to put in your files you want to host.. This is the pathname the software uses to
#define row "<tr> <th scope=\"row\">%s</th> <td>%d</td> </tr>"

typedef struct statistics {
	char *name;
	int num;

} statistics;

// read files from
int get_file_size(int fd)
{
	struct stat stat_struct;
	if(fstat(fd, &stat_struct) == -1)
		return(1);
	return (int)stat_struct.st_size;
}
void send_new(int fd,char *msg)
{
	int len = strlen(msg);
	if(send(fd,msg,len,0) == -1)
	{
		printf("Error in send\n");
	}
}



void sendlarge(char *fd, int sock_fd){
  uint8_t buff[4000];
  int actually_read;
  while((actually_read = read(fd, buff, sizeof(buff))) > 0)
  {
	  send(sock_fd, buff, actually_read, 0);
  }
}
int recv_new(int fd,char *buffer)	// This function recieves the buffer untill a "End of line" byte is recieved
{
#define EOL "\r\n"
#define EOL_SIZE 2
	char *p = buffer;	// we'll be using a pointer to the buffer than to mess with buffer directly
	int eol_matched=0;	// this is used to see that the recieved byte matched the buffer byte or not
	while(recv(fd,p,1,0)!=0)	// start recv bytes 1 by 1
	{
		if( *p == EOL[eol_matched])	// if the byte matched the first eol byte that is '\r'
		{
			++eol_matched;
			if(eol_matched==EOL_SIZE)	// if both the bytes matches the EOL
			{
				*(p+1-EOL_SIZE) = '\0';	// End the string
				return(strlen(buffer));	// return the bytes recieved
			}
		}
		else
		{
			eol_matched=0;
		}
		p++;	// increment the pointer to recv next byte
	}
	return(0);
}
int connection(int fd)
{
	char request[500000],resource[50000],*ptr;
	int fd1,length;
	if(recv_new(fd,request) == 0)
	{
		printf("Recieve failed\n");
	}
	// Checking for a valid browser request
	ptr = strstr(request," HTTP/");
	if(ptr == NULL)
	{
		printf("NOT HTTP!!\n");
	}
	else
	{
		*ptr=0;
		ptr=NULL;
		if( strncmp(request,"GET ",4) == 0)
		{
			ptr=request+4;
		}
		if(strncmp(request,"HEAD ",5) == 0)
			ptr=request+5;
		if(ptr==NULL)
		{
			printf("Unknown Request !!! \n");
		}
		else
		{
			if( ptr[strlen(ptr) - 1] == '/' )
			{
				strcat(ptr,"index.html");
			} 
			
			strcpy(resource,webroot);
			strcat(resource,ptr);
			fd1 = open(resource,O_RDONLY,0);
			printf("Opening \"%s\"\n",resource);
			if(fd1 == -1)
			{
				printf("404 File not found Error\n");
				send_new(fd,"HTTP/1.0 404 Not Found\r\n");
				send_new(fd,"Server : Aneesh/Private\r\n\r\n");
				send_new(fd,"<html><head><title>404 not found error </title></head></html>\r\n\r\n");
				
			}
			else
			{
				printf("200 OK!!!\n");
				send_new(fd,"HTTP/1.0 200 OK\r\n");
				send_new(fd,"Server : Aneesh/Private\r\n\r\n");
				if( ptr == request+4 ) // if it is a get request
				{
					if( (length = get_file_size(fd1)) == -1 )
					{
						printf("Error getting size \n");
					}
					if( (ptr = (char *)malloc(length) ) == NULL )
						printf("Error allocating memory!!\n");
					read(fd1,ptr,length);

					//bindhttpfile(ptr);

					//int aux = write (fd, ptr, strlen(ptr)+1);
					//sendlarge(ptr,fd);
					int aux = send(fd,ptr,length,0);
					if(aux == -1)
					{

						printf("Send err!!\n");
					}
					//printf("%s",ptr);
					//printf("=%d\n",aux);

					//printf("??%d\n",length);

					free(ptr);
				}
			}
			close(fd);
		}
	}
	shutdown(fd,SHUT_RDWR);
}




int main(int argc, char *argv[])
{
	int sockfd,newfd;
	int err;
	struct addrinfo *res,*p,hints;
	struct sockaddr_storage their_addr;
	socklen_t addr_size;
	int yes=1;
	char ip[INET6_ADDRSTRLEN];
	memset(&hints,0,sizeof(hints));
	hints.ai_family=AF_UNSPEC;
	hints.ai_flags=AI_PASSIVE;
	hints.ai_socktype=SOCK_STREAM;
	printf("Server is open for listening on port 80\n");
	if( (err = getaddrinfo(NULL,port,&hints,&res) ) == -1)
	{
		printf("Err in getaddrinfo : %s\n",gai_strerror(err));
		return(1);
	}
	for(p=res;p!=NULL;p=p->ai_next)
	{
		if( ( sockfd = socket(p->ai_family,p->ai_socktype,p->ai_protocol) ) == -1)
		{
			printf("Socket error !!!\n");
			continue;
		}
		if( setsockopt(sockfd,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(int)) == -1)
		{
			printf("Setsockopt err!!\n");
			return(1);
		}
		if( bind(sockfd,p->ai_addr,p->ai_addrlen) == -1)
		{
			printf("Binding err\n");
			close(sockfd);
			continue;
		}
		break;
	}
	if( listen(sockfd,15) == -1)
	{
		printf("Error in listen\n");
		return(1);
	}
	/*
	statistics *values;
	if( (values = (statistics *)malloc(argc) ) == NULL )
							printf("Error allocating memory!!\n");

	int i;
	for (i = 1;  i < argc; i++) {

		values[i].num=argv[i];

	}
	*/
	while(1)
	{
		char y;
		addr_size = sizeof(their_addr);
		if( ( newfd = accept(sockfd, (struct sockaddr *)&their_addr,&addr_size) ) == -1)
		{
			printf("Error in accept!\n");
			return(1);
		}
		for(p=res;p!=NULL;p=p->ai_next)
		{
			void *addr;
			if(p->ai_family == AF_INET)
			{
				struct sockaddr_in *ip;
				ip = (struct sockaddr_in *)p->ai_addr;
				addr = &(ip->sin_addr);
			}
			if(p->ai_family == AF_INET6)
			{
				struct sockaddr_in6 *ip;
				ip = (struct sockaddr_in6 *)p->ai_addr;
				addr = &(ip->sin6_addr);
			}
			inet_ntop(p->ai_family,addr,ip,sizeof(ip));
			printf("Got connection from %s\n",ip);
		}
		connection(newfd);
	}
	freeaddrinfo(res);
	close(newfd);
	close(sockfd);
	return(0);
}



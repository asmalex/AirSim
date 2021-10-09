// Fill out your copyright notice in the Description page of Project Settings.


#include "MyActor.h"
#include "Modules/ModuleManager.h"
#include "C:\Program Files\Epic Games\UE_4.26\Engine\Source\Runtime\Online\WebSockets\Public\WebSocketsModule.h" // Module definition
#include "C:\Program Files\Epic Games\UE_4.26\Engine\Source\Runtime\Online\WebSockets\Public\IWebSocket.h"       // Socket definition

TSharedPtr<IWebSocket> Socket;

// Sets default values
AMyActor::AMyActor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void AMyActor::BeginPlay()
{
    const FString ServerURL = TEXT("ws://127.0.0.1:30020/obstacle"); // Your server URL. You can use ws, wss or wss+insecure.
    const FString ServerProtocol = TEXT("ws"); // The WebServer protocol you want to use.

    AMyActor::goals = 0; // Number of obstacles that have been passed through in binary

    Socket = FWebSocketsModule::Get().CreateWebSocket(ServerURL, ServerProtocol);

    // We bind all available events
    Socket->OnConnected().AddLambda([]() -> void {
        // This code will run once connected.
        if (GEngine)
            GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Blue, TEXT("Object Websocket connected"));
    });

    // TODO: Program exits after failing to find client
    Socket->OnConnectionError().AddLambda([](const FString& Error) -> void {
        // This code will run if the connection failed. Check Error to see what happened.
        if (GEngine) {
            GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, *Error);
        }
    });

    Socket->OnClosed().AddLambda([](int32 StatusCode, const FString& Reason, bool bWasClean) -> void {
        // This code will run when the connection to the server has been terminated.
        // Because of an error or a call to Socket->Close().
        if (GEngine)
            GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Blue, TEXT("Object Websocket closed"));
        if (bWasClean)
            GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, TEXT("Object Websocket connection error! Severence from server was unclean"));
    });

    Socket->OnMessage().AddLambda([](const FString& Message) -> void {
        // This code will run when we receive a string message from the server.
        if (GEngine)
            GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Blue, TEXT("Message Received: String! printing message..."));
        if (GEngine)
            GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Blue, *Message);
    });

    Socket->OnRawMessage().AddLambda([](const void* Data, SIZE_T Size, SIZE_T BytesRemaining) -> void {
        // This code will run when we receive a raw (binary) message from the server.
        if (GEngine)
            GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Blue, TEXT("Message Received: Binary!"));
    });

    Socket->OnMessageSent().AddLambda([](const FString& MessageString) -> void {
        // This code is called after we sent a message to the server.
        if (GEngine)
            GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Blue, TEXT("Sending Message: " + MessageString));
    });

    // And we finally connect to the server.
    Socket->Connect();

}

// Called every frame
void AMyActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

// Increments the number of obstacles passed through and sends the value through the WebSocket
void AMyActor::SendMessageSocket()
{
    if (!Socket->IsConnected()) {
        // Don't send if we're not connected.
        return;
    }

    AMyActor::goals = AMyActor::goals++;
    int temp = AMyActor::goals;

    Socket->Send(&temp, sizeof(char));
}

void AMyActor::ClosePort()
{
    Socket->Close();
}

// STORAGE

//  GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, TEXT("Object Websocket connection error! Severence from server was unclean"));
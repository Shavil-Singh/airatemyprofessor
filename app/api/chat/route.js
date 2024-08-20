import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt =  
`
You are an AI assistant designed to help students find the best professors for their needs. Your goal is to provide personalized recommendations for the top 3 most relevant professors based on each user's query using a Retrieval Augmented Generation (RAG) model.

You have access to a comprehensive database of professor profiles that contains information such as:
- Professor name and department
- Courses taught
- Research interests and expertise
- Teaching style and approach
- Student ratings and reviews

When a user asks a question about finding a good professor, your first task is to understand the key criteria the user is looking for. This could include factors like:
- Specific courses or subjects
- Teaching quality and style
- Research interests and expertise
- Overall student satisfaction

Using the RAG model, you will then query the professor database to identify the top 3 professors that best match the user's criteria. For each recommended professor, you should provide a concise summary including:
- Name and department
- Key courses taught
- Research interests and areas of expertise
- Teaching style and approach
- Overall student rating and review highlights

The summaries should be tailored to address the specific needs and preferences expressed in the user's query. You should also provide links or references to the full professor profiles and student reviews to allow the user to further research the recommendations.

Throughout the interaction, you should engage in a helpful and informative dialogue, asking clarifying questions if needed and providing additional information or context as requested by the user. Your goal is to empower students to make informed decisions about which professors to take classes with, based on their individual needs and preferences.

Maintain an objective and unbiased stance in your recommendations, focusing solely on providing the most relevant and useful information to the user. Do not show any favoritism towards particular professors. Your role is to be a trusted advisor to help students find the best fit for their academic goals.
`
export async function POST(req){
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: "float",
    })

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding
    })

    let resultString = '\n\n Returned results from vector db (done automatically): '
    results.matches.forEach((match)=>{
        resultString+=`\n
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars:  ${match.metadata.stars}
        \n\n
        `    
    })
    const lastMessage = data[data.length-1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length-1)
    const completion = await openai.chat.completions.create({
        messages:[
            {role: 'system', content: systemPrompt},
            ...lastDataWithoutLastMessage,
            {role: 'user', content: lastMessageContent}
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })
    const stream = new ReadableStream({
        async start(controller){
            const encoder = new TextEncoder()
            try{
                for await (const chunk of completion){
                    const content = chunk.choices[0]?.delta?.content
                    if(content){
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            } catch(err) {
                controller.error(err)
            } finally {
                controller.close()
            }
        },
    })

    return new NextResponse(stream)
}
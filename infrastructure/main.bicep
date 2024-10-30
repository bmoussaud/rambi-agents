//
// Sure! Here are the regions that support the 'Standard' SKU for the 'gpt-4o 2024-08-06' model:
//East US
//East US 2
//North Central US
//Sweden Central123
//
@description('Location of the resources')
param location string = resourceGroup().location

@description('Model deployments for OpenAI')
param deployments array = [
  {
    name: 'gpt-4o'
    capacity: 40
    version: '2024-05-13'
  }
  {
    name: 'text-embedding-ada-002'
    capacity: 120
    version: '2'
  }
]

@description('Restore the service instead of creating a new instance. This is useful if you previously soft-deleted the service and want to restore it. If you are restoring a service, set this to true. Otherwise, leave this as false.')
param restore bool = false

var prefix = uniqueString(resourceGroup().id)
var searchServiceName = '${prefix}-search'
var openAIName = '${prefix}-openai'

@description('Creates an Azure AI Search service.')
resource searchService 'Microsoft.Search/searchServices@2023-11-01' = {
  name: searchServiceName
  location: location
  sku: {
    name: 'standard'
  }
}

output searchServiceName string = searchService.name
output searchServiceStatus string = searchService.properties.status
output searchServiceAdminKey string = listAdminKeys(searchService.id, '2020-08-01').primaryKey
output searchServiceEndpoint string = 'https://${searchService.name}.search.windows.net'

@description('Creates an Azure OpenAI resource.')
resource openAI 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: openAIName
  location: location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: openAIName
    publicNetworkAccess: 'Enabled'
    restore: restore
  }
}

output cognitiveAccountName string = openAI.name
output cognitiveAccountEndpoint string = openAI.properties.endpoint
output cognitiveAccountKey string = listKeys(openAI.id, '2021-04-30').key1

// Create the Openai model deployments
resource gpt4oEndpoint 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openAI
  name: 'gpt4o'

  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o'
      version: '2024-08-06'
    }
  }
}

class Vehicle(BaseModel):
  make: str = Field(..., description="The make of the vehicle")
  model: str = Field(..., description="The model of the vehicle")
  motor: Literal['essence', 'diesel', 'hybride', 'hybride rechargeable', 'électrique'] = Field(..., description="The fuel / motor type of the vehicle")

class Insurance(BaseModel):
  type: Literal['omnium', 'mini-omnium', 'rc', 'aucune'] = Field(..., description="The type of insurance; 'aucune' if there is no insurance")
  company: str = Field(..., description="The insurance company; empty string if insurance is 'rc' only or 'aucune'")
  monthly_prime: float = Field(..., description="The separate monthly insurance premium, in euros; 0 if there is no separate premium")

class Contract(BaseModel):
  contract_number: str = Field(..., description="The contract number, e.g. ALW-2026-1914")
  contract_date: date = Field(..., description="The date of the contract. Format YYYY-MM-DD, whatever the format on the document")
  insurance: Insurance = Field(..., description="The insurance covering the leasing contract")
  vehicle: Vehicle = Field(..., description="The vehicle being leased")
  agency: str = Field(..., description="The name of the leasing company")
  client: str = Field(..., description="The name of the customer (or 'preneur'), NOT the sales advisor")
  duration: int = Field(..., description="Duration of the contract, in months")
  annual_km: int = Field(..., description="The maximum amount of kilometers per year that the contract allows")
  deposit: float = Field(..., description="Deposit amount, in euros; 0 if there is none")
  monthly_amount: float = Field(..., description="Monthly payment amount, VAT included (TVAC); if the amount was corrected by hand, the corrected amount")
  client_signature: bool = Field(..., description="True only if a handwritten signature of the client is present")
  agency_signature: bool = Field(..., description="True only if a handwritten signature of the agency is present")
  cancel: bool = Field(..., description="True if the document bears a cancellation mark (e.g. 'ANNULÉ' stamp), not just a clause explaining how to cancel")
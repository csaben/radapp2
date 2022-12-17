pragma solidity ^0.8.15;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/utils/Strings.sol";

// Chainlink Imports
import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";
import "@chainlink/contracts/src/v0.8/KeeperCompatible.sol";

// Dev imports. This only works on a local dev network
// and will not work on any test or main livenets.
import "hardhat/console.sol";

contract DicomNFT is ERC721, ERC721Enumerable, ERC721URIStorage, Ownable {
    using Counters for Counters.Counter;

    Counters.Counter private _tokenIdCounter;

    // Filebase client contract address
    FilebaseClient private _filebaseClient;

    constructor(FilebaseClient filebaseClient) ERC721("Dicom NFT", "DCM") {
        _filebaseClient = filebaseClient;
    }

    function safeMint(address to, bytes32 filebaseFileId) public {
        // Current counter value will be the minted token's token ID.
        uint256 tokenId = _tokenIdCounter.current();

        // Increment it so next time it's correct when we call .current()
        _tokenIdCounter.increment();

        // Mint the token
        _safeMint(to, tokenId);

        // Set the URI of the NFT to the filebase file ID
        string memory filebaseFileIdStr = bytes32ToString(filebaseFileId);
        _setTokenURI(tokenId, filebaseFileIdStr);

        console.log(
            "DONE!!! minted token ",
            tokenId,
            " and assigned token filebase file ID: ",
            filebaseFileIdStr
        );
    }

    function getMetadataUri(uint256 tokenId) public view
